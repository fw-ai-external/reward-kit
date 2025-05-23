import unittest
from typing import Any, Dict, List, Union, cast  # Added cast, Union, List, Dict, Any

from examples.bfcl_agent_example.envs.gorilla_file_system import (
    Directory,
    File,
    GorillaFileSystem,
)
from examples.bfcl_agent_example.envs.math_api import MathAPI
from examples.bfcl_agent_example.envs.message_api import MessageAPI
from examples.bfcl_agent_example.envs.posting_api import TwitterAPI
from examples.bfcl_agent_example.envs.ticket_api import TicketAPI
from examples.bfcl_agent_example.envs.trading_bot import TradingBot
from examples.bfcl_agent_example.envs.travel_booking import TravelAPI
from examples.bfcl_agent_example.envs.vehicle_control import VehicleControlAPI

# Import other tool APIs as they are tested


class TestBfclTools(unittest.TestCase):

    def test_math_api_instantiation(self):
        """Test that MathAPI can be instantiated."""
        try:
            math_tool = MathAPI()
            self.assertIsNotNone(math_tool)
        except Exception as e:
            self.fail(f"MathAPI instantiation failed: {e}")

    def test_math_api_add(self):
        """Test MathAPI's add method."""
        math_tool = MathAPI()
        result = math_tool.add(a=5.0, b=3.0)  # Changed to float
        self.assertEqual(result.get("result"), 8)
        # Pass a type that will cause an error in the method if not handled
        result_error = math_tool.add(
            a=cast(float, "test"), b=3.0
        )  # Cast to float, will error
        self.assertIn("error", result_error)

    # Add more tests for MathAPI methods here

    def test_message_api_instantiation_and_load_scenario(self):
        """Test MessageAPI instantiation and _load_scenario."""
        try:
            message_tool = MessageAPI()
            self.assertIsNotNone(message_tool)
            message_tool._load_scenario({})
            self.assertIsNotNone(getattr(message_tool, "user_map", None))
        except Exception as e:
            self.fail(f"MessageAPI instantiation or _load_scenario failed: {e}")

    def test_message_api_list_users(self):
        """Test MessageAPI's list_users method."""
        message_tool = MessageAPI()
        message_tool._load_scenario({})
        result = message_tool.list_users()
        self.assertIn("user_list", result)
        self.assertIsInstance(result["user_list"], list)
        self.assertEqual(len(result["user_list"]), 4)

    def test_ticket_api_instantiation_and_load_scenario(self):
        """Test TicketAPI instantiation and _load_scenario."""
        try:
            ticket_tool = TicketAPI()
            self.assertIsNotNone(ticket_tool)
            ticket_tool._load_scenario({})
            self.assertIsNotNone(getattr(ticket_tool, "ticket_queue", None))
        except Exception as e:
            self.fail(f"TicketAPI instantiation or _load_scenario failed: {e}")

    def test_ticket_api_get_ticket(self):
        """Test TicketAPI's get_ticket method."""
        ticket_tool = TicketAPI()
        ticket_tool._load_scenario({})

        result_error = ticket_tool.get_ticket(ticket_id=999)
        self.assertIn("error", result_error)

        ticket_tool.current_user = "test_user"
        create_result = ticket_tool.create_ticket(
            title="Test Ticket", description="A test ticket"
        )
        ticket_id_any = create_result.get("id")
        self.assertIsNotNone(ticket_id_any)
        if ticket_id_any is not None:  # Type guard
            ticket_id = int(ticket_id_any)  # Ensure it's an int
            get_result = ticket_tool.get_ticket(ticket_id=ticket_id)
            self.assertEqual(get_result.get("id"), ticket_id)
            self.assertEqual(get_result.get("title"), "Test Ticket")
        else:
            self.fail("Ticket ID was None after creation")

    def test_trading_bot_instantiation_and_load_scenario(self):
        """Test TradingBot instantiation and _load_scenario."""
        try:
            trading_tool = TradingBot()
            self.assertIsNotNone(trading_tool)
            trading_tool._load_scenario({})
            self.assertIsNotNone(getattr(trading_tool, "orders", None))
            self.assertIsNotNone(getattr(trading_tool, "stocks", None))
        except Exception as e:
            self.fail(f"TradingBot instantiation or _load_scenario failed: {e}")

    def test_trading_bot_get_stock_info(self):
        """Test TradingBot's get_stock_info method."""
        trading_tool = TradingBot()
        trading_tool._load_scenario({})

        result_aapl = trading_tool.get_stock_info(symbol="AAPL")
        self.assertNotIn("error", result_aapl)
        self.assertIn("price", result_aapl)

        result_error = trading_tool.get_stock_info(symbol="NONEXISTENT")
        self.assertIn("error", result_error)

    def test_travel_api_instantiation_and_load_scenario(self):
        """Test TravelAPI instantiation and _load_scenario."""
        try:
            travel_tool = TravelAPI()
            self.assertIsNotNone(travel_tool)
            travel_tool._load_scenario({})
            self.assertIsNotNone(getattr(travel_tool, "credit_card_list", None))
            self.assertIsNotNone(getattr(travel_tool, "booking_record", None))
        except Exception as e:
            self.fail(f"TravelAPI instantiation or _load_scenario failed: {e}")

    def test_travel_api_get_flight_cost(self):
        """Test TravelAPI's get_flight_cost method."""
        travel_tool = TravelAPI()
        travel_tool._load_scenario({})

        result = travel_tool.get_flight_cost(
            travel_from="SFO",
            travel_to="LAX",
            travel_date="2024-10-01",
            travel_class="economy",
        )
        self.assertIn("travel_cost_list", result)
        travel_cost_list = result["travel_cost_list"]
        if isinstance(travel_cost_list, list):  # Type guard
            self.assertTrue(len(travel_cost_list) > 0)
            self.assertIsInstance(travel_cost_list[0], float)
        else:
            self.fail("travel_cost_list is not a list")

        with self.assertRaises(ValueError) as context:
            travel_tool.get_flight_cost(
                travel_from="SFO",
                travel_to="LAX",
                travel_date="2024-10-01",
                travel_class="invalid_class",
            )
        self.assertIn("Invalid travel class", str(context.exception))

        with self.assertRaises(ValueError) as context:
            travel_tool.get_flight_cost(
                travel_from="XXX",
                travel_to="YYY",
                travel_date="2024-10-01",
                travel_class="economy",
            )
        self.assertIn("No available route", str(context.exception))

    def test_vehicle_control_api_instantiation_and_load_scenario(self):
        """Test VehicleControlAPI instantiation and _load_scenario."""
        try:
            vehicle_tool = VehicleControlAPI()
            self.assertIsNotNone(vehicle_tool)
            vehicle_tool._load_scenario({})
            self.assertIsNotNone(getattr(vehicle_tool, "fuelLevel", None))
        except Exception as e:
            self.fail(f"VehicleControlAPI instantiation or _load_scenario failed: {e}")

    def test_vehicle_control_api_display_car_status(self):
        """Test VehicleControlAPI's displayCarStatus method."""
        vehicle_tool = VehicleControlAPI()
        vehicle_tool._load_scenario({})

        result_fuel = vehicle_tool.displayCarStatus(option="fuel")
        self.assertIn("fuelLevel", result_fuel)
        self.assertIsInstance(result_fuel["fuelLevel"], float)

        result_error = vehicle_tool.displayCarStatus(option="invalid_option")
        self.assertIn("error", result_error)

    def test_gorilla_file_system_instantiation_and_load_scenario(self):
        """Test GorillaFileSystem instantiation and _load_scenario."""
        try:
            fs_tool = GorillaFileSystem()
            self.assertIsNotNone(fs_tool)
            fs_tool._load_scenario({})
            self.assertIsNotNone(getattr(fs_tool, "root", None))
        except Exception as e:
            self.fail(f"GorillaFileSystem instantiation or _load_scenario failed: {e}")

    def test_gorilla_file_system_ls_and_cat(self):
        """Test GorillaFileSystem's ls and cat methods."""
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario(
            {
                "root": {
                    "type": "directory",
                    "contents": {
                        "example.txt": {"type": "file", "content": "test content"}
                    },
                }
            }
        )

        ls_result = fs_tool.ls()
        self.assertIn("contents", ls_result)
        ls_contents = ls_result["contents"]
        if isinstance(ls_contents, dict):  # Type guard
            self.assertIn("example.txt", ls_contents)
            self.assertEqual(ls_contents["example.txt"]["type"], "file")
        else:
            self.fail("ls_result['contents'] is not a dict")

        cat_result = fs_tool.cat(file_name="example.txt")
        self.assertEqual(cat_result.get("content"), "test content")

        cat_error_result = fs_tool.cat(file_name="non_existent_file.txt")
        self.assertIn("error", cat_error_result)

    def test_twitter_api_instantiation_and_load_scenario(self):
        """Test TwitterAPI instantiation and _load_scenario."""
        try:
            twitter_tool = TwitterAPI()
            self.assertIsNotNone(twitter_tool)
            twitter_tool._load_scenario({})
            self.assertIsNotNone(getattr(twitter_tool, "tweets", None))
            self.assertFalse(twitter_tool.authenticated)
        except Exception as e:
            self.fail(f"TwitterAPI instantiation or _load_scenario failed: {e}")

    def test_twitter_api_post_and_get_tweets(self):
        """Test TwitterAPI's post_tweet and get_tweets methods."""
        twitter_tool = TwitterAPI()
        scenario_data = {
            "username": "testuser",
            "password": "password",
            "tweets": {},
            "comments": {},
            "retweets": {},
            "following_list": [],
            "tweet_counter": 0,
            "authenticated": False,
        }
        twitter_tool._load_scenario(scenario_data)

        login_result = twitter_tool.login(username="testuser", password="password")
        self.assertEqual(
            login_result.get("status"),
            "success",
            f"Login failed: {login_result.get('message')}",
        )
        self.assertTrue(twitter_tool.authenticated)

        tweet_content = "Hello from Cline the AI!"
        post_result = twitter_tool.post_tweet(content=tweet_content)
        self.assertEqual(
            post_result.get("status"),
            "success",
            f"Post tweet failed: {post_result.get('message')}",
        )
        self.assertIn("tweet_id", post_result)
        tweet_id = post_result["tweet_id"]
        tweet_id_str = str(tweet_id)

        tweets_result = twitter_tool.get_tweets(username="testuser")
        self.assertIsInstance(
            tweets_result, dict, "get_tweets should return a dictionary."
        )
        self.assertIn(
            tweet_id_str, tweets_result, "Posted tweet not found in user's tweets dict."
        )
        self.assertEqual(tweets_result[tweet_id_str]["content"], tweet_content)
        self.assertEqual(tweets_result[tweet_id_str]["username"], "testuser")

        all_tweets_result = twitter_tool.get_tweets()
        self.assertIsInstance(all_tweets_result, dict)
        self.assertIn(
            tweet_id_str,
            all_tweets_result,
            "Posted tweet not found in all tweets dict.",
        )
        self.assertEqual(all_tweets_result[tweet_id_str]["content"], tweet_content)

    def test_gorilla_file_system_pwd(self):
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario({})
        pwd_result_root = fs_tool.pwd()
        self.assertNotIn(
            "error",
            pwd_result_root,
            f"pwd failed at root: {pwd_result_root.get('error')}",
        )
        self.assertEqual(pwd_result_root.get("current_working_directory"), "/workspace")
        fs_tool.mkdir(dir_name="testdir")
        fs_tool.cd(folder="testdir")
        pwd_result_subdir = fs_tool.pwd()
        self.assertNotIn(
            "error",
            pwd_result_subdir,
            f"pwd failed in subdir: {pwd_result_subdir.get('error')}",
        )
        self.assertEqual(
            pwd_result_subdir.get("current_working_directory"), "/workspace/testdir"
        )
        fs_tool.cd(folder="..")
        pwd_result_back_to_root = fs_tool.pwd()
        self.assertNotIn(
            "error",
            pwd_result_back_to_root,
            f"pwd failed after cd ..: {pwd_result_back_to_root.get('error')}",
        )
        self.assertEqual(
            pwd_result_back_to_root.get("current_working_directory"), "/workspace"
        )

    def test_gorilla_file_system_cp(self):
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario(
            {
                "root": {
                    "type": "directory",
                    "contents": {
                        "file1.txt": {"type": "file", "content": "content of file1"},
                        "dir1": {
                            "type": "directory",
                            "contents": {
                                "nested_file.txt": {
                                    "type": "file",
                                    "content": "nested content",
                                }
                            },
                        },
                    },
                }
            }
        )

        cp_result1 = fs_tool.cp(source="file1.txt", destination="file2.txt")
        self.assertIn("Copied 'file1.txt' to 'file2.txt'", cp_result1.get("result", ""))
        self.assertIn("file2.txt", fs_tool.current_dir.contents)
        file2_node = fs_tool.current_dir.contents["file2.txt"]
        if isinstance(file2_node, File):
            self.assertEqual(file2_node.content, "content of file1")
        else:
            self.fail("file2.txt is not a File")

        cp_result2 = fs_tool.cp(source="file1.txt", destination="dir1")
        self.assertIn(
            "Copied 'file1.txt' into directory 'dir1'", cp_result2.get("result", "")
        )
        dir1_node = fs_tool.current_dir.contents["dir1"]
        if isinstance(dir1_node, Directory):
            self.assertIn("file1.txt", dir1_node.contents)
            copied_file_in_dir1 = dir1_node.contents["file1.txt"]
            if isinstance(copied_file_in_dir1, File):
                self.assertEqual(copied_file_in_dir1.content, "content of file1")
            else:
                self.fail("Copied file1.txt in dir1 is not a File")
        else:
            self.fail("dir1 is not a Directory")

        cp_result3 = fs_tool.cp(source="dir1", destination="dir2")
        self.assertIn("Copied 'dir1' to 'dir2'", cp_result3.get("result", ""))
        self.assertIn("dir2", fs_tool.current_dir.contents)
        dir1_original_ref = fs_tool.current_dir.contents.get(
            "dir1"
        )  # dir1 might be gone if source was 'dir1' and dest was 'dir2' (rename)
        # If cp is copy, dir1 should still exist.
        # Assuming cp is copy, not move.
        dir2_copied = fs_tool.current_dir.contents["dir2"]

        if isinstance(dir2_copied, Directory) and isinstance(
            dir1_original_ref, Directory
        ):
            self.assertIsInstance(dir2_copied, type(dir1_original_ref))
            self.assertIn("nested_file.txt", dir2_copied.contents)
            nested_file_in_dir2 = dir2_copied.contents["nested_file.txt"]
            if isinstance(nested_file_in_dir2, File):
                self.assertEqual(nested_file_in_dir2.content, "nested content")
                nested_file_in_dir2.content = "modified nested"  # Modify copy
            else:
                self.fail("nested_file.txt in dir2 is not a File")

            original_nested_file_in_dir1 = dir1_original_ref.contents["nested_file.txt"]
            if isinstance(original_nested_file_in_dir1, File):
                self.assertEqual(
                    original_nested_file_in_dir1.content, "nested content"
                )  # Check original is unchanged
            else:
                self.fail("Original nested_file.txt in dir1 is not a File")
        elif not isinstance(dir1_original_ref, Directory):
            self.fail(
                "Original dir1 reference is not a Directory or was removed unexpectedly"
            )
        else:
            self.fail("dir2 is not a Directory")

        fs_tool.mkdir("dir3")
        cp_result4 = fs_tool.cp(source="dir1", destination="dir3")
        self.assertIn(
            "Copied 'dir1' into directory 'dir3'", cp_result4.get("result", "")
        )
        dir3_node = fs_tool.current_dir.contents["dir3"]
        if isinstance(dir3_node, Directory):
            self.assertIn("dir1", dir3_node.contents)
            copied_dir1_in_dir3 = dir3_node.contents["dir1"]
            if isinstance(copied_dir1_in_dir3, Directory):
                self.assertIn("nested_file.txt", copied_dir1_in_dir3.contents)
            else:
                self.fail("Copied dir1 in dir3 is not a Directory")
        else:
            self.fail("dir3 is not a Directory")

        cp_result5 = fs_tool.cp(source="nonexistent.txt", destination="newfile.txt")
        self.assertIn(
            "Error: Source 'nonexistent.txt' not found", cp_result5.get("result", "")
        )

        fs_tool.mkdir("dir_target_for_file")
        cp_result6 = fs_tool.cp(source="file1.txt", destination="dir_target_for_file")
        self.assertIn(
            "Copied 'file1.txt' into directory 'dir_target_for_file'",
            cp_result6.get("result", ""),
        )
        dir_target_node = fs_tool.current_dir.contents["dir_target_for_file"]
        if isinstance(dir_target_node, Directory):
            self.assertIn("file1.txt", dir_target_node.contents)
            copied_file_in_dir_target = dir_target_node.contents["file1.txt"]
            if isinstance(copied_file_in_dir_target, File):
                self.assertEqual(copied_file_in_dir_target.content, "content of file1")
            else:
                self.fail("Copied file1.txt in dir_target_for_file is not a File")
        else:
            self.fail("dir_target_for_file is not a Directory")

        fs_tool.current_dir.contents["file_to_be_overwritten_by_dir.txt"] = File(
            name="file_to_be_overwritten_by_dir.txt",
            content="original file content",
            parent=fs_tool.current_dir,
        )
        if "dir1" not in fs_tool.current_dir.contents:  # Ensure dir1 exists
            fs_tool.mkdir("dir1")
        cp_result7 = fs_tool.cp(
            source="dir1", destination="file_to_be_overwritten_by_dir.txt"
        )
        self.assertIn(
            "Error: Cannot overwrite file 'file_to_be_overwritten_by_dir.txt' with directory 'dir1'",
            cp_result7.get("result", ""),
        )

        fs_tool.current_dir.contents["file_to_overwrite.txt"] = File(
            name="file_to_overwrite.txt",
            content="original content",
            parent=fs_tool.current_dir,
        )
        cp_result8 = fs_tool.cp(source="file1.txt", destination="file_to_overwrite.txt")
        self.assertIn(
            "Copied 'file1.txt' to 'file_to_overwrite.txt'",
            cp_result8.get("result", ""),
        )
        file_to_overwrite_node = fs_tool.current_dir.contents["file_to_overwrite.txt"]
        if isinstance(file_to_overwrite_node, File):
            self.assertEqual(file_to_overwrite_node.content, "content of file1")
        else:
            self.fail("file_to_overwrite.txt is not a File after cp")

        fs_tool.mkdir("dir_for_overwrite_test")
        dir_for_overwrite_node = fs_tool.current_dir.contents["dir_for_overwrite_test"]
        if isinstance(dir_for_overwrite_node, Directory):
            dir_for_overwrite_node.contents["common_name.txt"] = File(
                name="common_name.txt",
                content="content in dir",
                parent=dir_for_overwrite_node,
            )
        else:
            self.fail("dir_for_overwrite_test is not a Directory")

        fs_tool.current_dir.contents["common_name.txt"] = File(
            name="common_name.txt",
            content="new main content",
            parent=fs_tool.current_dir,
        )
        cp_result9 = fs_tool.cp(
            source="common_name.txt", destination="dir_for_overwrite_test"
        )
        self.assertIn(
            "Copied 'common_name.txt' into directory 'dir_for_overwrite_test'",
            cp_result9.get("result", ""),
        )

        target_dir_node_after_cp = fs_tool.current_dir.contents[
            "dir_for_overwrite_test"
        ]
        if isinstance(target_dir_node_after_cp, Directory):
            overwritten_file_node = target_dir_node_after_cp.contents["common_name.txt"]
            if isinstance(overwritten_file_node, File):
                self.assertEqual(overwritten_file_node.content, "new main content")
            else:
                self.fail(
                    "common_name.txt in dir_for_overwrite_test is not a File after cp"
                )
        else:
            self.fail("dir_for_overwrite_test is not a Directory after cp")

    def test_gorilla_file_system_diff(self):
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario({})
        fs_tool.current_dir.contents["file_a.txt"] = File(
            name="file_a.txt", content="line1\nline2\nline3", parent=fs_tool.current_dir
        )
        fs_tool.current_dir.contents["file_b.txt"] = File(
            name="file_b.txt", content="line1\nline2\nline3", parent=fs_tool.current_dir
        )
        fs_tool.current_dir.contents["file_c.txt"] = File(
            name="file_c.txt",
            content="line_one\nline2\nline_three\nline4",
            parent=fs_tool.current_dir,
        )

        diff_result1 = fs_tool.diff(file_name1="file_a.txt", file_name2="file_b.txt")
        self.assertEqual(diff_result1.get("status"), "success")
        self.assertEqual(diff_result1.get("diff_lines"), "Files are identical")

        diff_result2 = fs_tool.diff(file_name1="file_a.txt", file_name2="file_c.txt")
        self.assertEqual(diff_result2.get("status"), "success")
        expected_diff_lines = [
            "Line 1: 'line1' != 'line_one'",
            "Line 3: 'line3' != 'line_three'",
            "Line 4: None != 'line4'",
        ]
        self.assertEqual(diff_result2.get("diff_lines"), "\n".join(expected_diff_lines))

        diff_result2b = fs_tool.diff(file_name1="file_c.txt", file_name2="file_a.txt")
        self.assertEqual(diff_result2b.get("status"), "success")
        expected_diff_lines_rev = [
            "Line 1: 'line_one' != 'line1'",
            "Line 3: 'line_three' != 'line3'",
            "Line 4: 'line4' != None",
        ]
        self.assertEqual(
            diff_result2b.get("diff_lines"), "\n".join(expected_diff_lines_rev)
        )

        diff_result3 = fs_tool.diff(
            file_name1="non_existent.txt", file_name2="file_a.txt"
        )
        self.assertIn("error", diff_result3)
        self.assertEqual(diff_result3.get("error"), "File non_existent.txt not found")

        diff_result4 = fs_tool.diff(
            file_name1="file_a.txt", file_name2="non_existent.txt"
        )
        self.assertIn("error", diff_result4)
        self.assertEqual(diff_result4.get("error"), "File non_existent.txt not found")

    def test_gorilla_file_system_touch(self):
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario({})
        touch_result1 = fs_tool.touch(file_name="new_empty_file.txt")
        self.assertEqual(touch_result1, {})
        self.assertIn("new_empty_file.txt", fs_tool.current_dir.contents)
        new_file_node = fs_tool.current_dir.contents["new_empty_file.txt"]
        if isinstance(new_file_node, File):
            self.assertEqual(new_file_node.content, "")
        else:
            self.fail("new_empty_file.txt is not a File")

        fs_tool.current_dir.contents["existing_file.txt"] = File(
            name="existing_file.txt", content="some content", parent=fs_tool.current_dir
        )
        touch_result2 = fs_tool.touch(file_name="existing_file.txt")
        self.assertEqual(touch_result2, {})
        existing_file_node = fs_tool.current_dir.contents["existing_file.txt"]
        if isinstance(existing_file_node, File):
            self.assertEqual(existing_file_node.content, "some content")
        else:
            self.fail("existing_file.txt is not a File")

        fs_tool.mkdir("existing_dir")
        touch_result3 = fs_tool.touch(file_name="existing_dir")
        self.assertIn("error", touch_result3)
        self.assertEqual(
            touch_result3.get("error"),
            "Cannot touch 'existing_dir': It is a directory.",
        )

    def test_gorilla_file_system_rm(self):
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario(
            {
                "root": {
                    "type": "directory",
                    "contents": {
                        "file_to_remove.txt": {"type": "file", "content": "content"},
                        "empty_dir_to_remove": {"type": "directory", "contents": {}},
                        "non_empty_dir_to_remove": {
                            "type": "directory",
                            "contents": {
                                "inner_file.txt": {
                                    "type": "file",
                                    "content": "inner content",
                                }
                            },
                        },
                    },
                }
            }
        )
        rm_result1 = fs_tool.rm(file_name="file_to_remove.txt")
        self.assertEqual(
            rm_result1.get("result"), "Successfully removed 'file_to_remove.txt'."
        )
        self.assertNotIn("file_to_remove.txt", fs_tool.current_dir.contents)
        rm_result2 = fs_tool.rm(file_name="empty_dir_to_remove")
        self.assertEqual(
            rm_result2.get("result"), "Successfully removed 'empty_dir_to_remove'."
        )
        self.assertNotIn("empty_dir_to_remove", fs_tool.current_dir.contents)
        rm_result3 = fs_tool.rm(file_name="non_empty_dir_to_remove")
        self.assertEqual(
            rm_result3.get("result"), "Successfully removed 'non_empty_dir_to_remove'."
        )
        self.assertNotIn("non_empty_dir_to_remove", fs_tool.current_dir.contents)
        rm_result4 = fs_tool.rm(file_name="does_not_exist.txt")
        self.assertEqual(
            rm_result4.get("result"), "Error: 'does_not_exist.txt' not found."
        )

    def test_gorilla_file_system_rmdir(self):
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario(
            {
                "root": {
                    "type": "directory",
                    "contents": {
                        "empty_dir": {"type": "directory", "contents": {}},
                        "non_empty_dir": {
                            "type": "directory",
                            "contents": {
                                "some_file.txt": {"type": "file", "content": "hello"}
                            },
                        },
                        "a_file.txt": {"type": "file", "content": "i am a file"},
                    },
                }
            }
        )
        rmdir_result1 = fs_tool.rmdir(dir_name="empty_dir")
        self.assertEqual(
            rmdir_result1.get("result"), "Successfully removed directory 'empty_dir'."
        )
        self.assertNotIn("empty_dir", fs_tool.current_dir.contents)
        rmdir_result2 = fs_tool.rmdir(dir_name="non_empty_dir")
        self.assertEqual(
            rmdir_result2.get("result"),
            "Error: Directory 'non_empty_dir' is not empty.",
        )
        self.assertIn("non_empty_dir", fs_tool.current_dir.contents)
        rmdir_result3 = fs_tool.rmdir(dir_name="a_file.txt")
        self.assertEqual(
            rmdir_result3.get("result"), "Error: 'a_file.txt' is not a directory."
        )
        self.assertIn("a_file.txt", fs_tool.current_dir.contents)
        rmdir_result4 = fs_tool.rmdir(dir_name="ghost_dir")
        self.assertEqual(
            rmdir_result4.get("result"), "Error: Directory 'ghost_dir' not found."
        )

    def test_gorilla_file_system_grep(self):
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario({})
        file_content = "Hello world\nThis is a test line\nAnother Test Line with test pattern\nworld wide web"
        fs_tool.current_dir.contents["grep_test_file.txt"] = File(
            name="grep_test_file.txt", content=file_content, parent=fs_tool.current_dir
        )
        grep_result1 = fs_tool.grep(file_name="grep_test_file.txt", pattern="world")
        self.assertNotIn("error", grep_result1)
        self.assertEqual(
            grep_result1.get("matching_lines"), ["Hello world", "world wide web"]
        )
        grep_result2 = fs_tool.grep(
            file_name="grep_test_file.txt", pattern="nonexistentpattern"
        )
        self.assertNotIn("error", grep_result2)
        self.assertEqual(grep_result2.get("matching_lines"), [])
        grep_result3 = fs_tool.grep(file_name="no_such_file.txt", pattern="world")
        self.assertIn("error", grep_result3)
        self.assertEqual(
            grep_result3.get("error"),
            "File 'no_such_file.txt' not found or is not a file.",
        )
        grep_result4 = fs_tool.grep(file_name="grep_test_file.txt", pattern="test")
        self.assertNotIn("error", grep_result4)
        self.assertEqual(
            grep_result4.get("matching_lines"),
            ["This is a test line", "Another Test Line with test pattern"],
        )
        grep_result5 = fs_tool.grep(file_name="grep_test_file.txt", pattern="Test")
        self.assertNotIn("error", grep_result5)
        self.assertEqual(
            grep_result5.get("matching_lines"), ["Another Test Line with test pattern"]
        )
        fs_tool.mkdir("grep_dir_test")
        grep_result6 = fs_tool.grep(file_name="grep_dir_test", pattern="world")
        self.assertIn("error", grep_result6)
        self.assertEqual(
            grep_result6.get("error"),
            "File 'grep_dir_test' not found or is not a file.",
        )

    def test_gorilla_file_system_mv(self):
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario(
            {
                "root": {
                    "type": "directory",
                    "contents": {
                        "file1.txt": {"type": "file", "content": "content1"},
                        "dir1": {"type": "directory", "contents": {}},
                        "dir2": {
                            "type": "directory",
                            "contents": {
                                "nested_file.txt": {"type": "file", "content": "nested"}
                            },
                        },
                        "file_to_overwrite.txt": {
                            "type": "file",
                            "content": "original_overwrite_content",
                        },
                    },
                }
            }
        )
        mv_result1 = fs_tool.mv(source="file1.txt", destination="file_renamed.txt")
        self.assertEqual(
            mv_result1.get("result"), "Moved 'file1.txt' to 'file_renamed.txt'."
        )
        self.assertNotIn("file1.txt", fs_tool.current_dir.contents)
        self.assertIn("file_renamed.txt", fs_tool.current_dir.contents)
        renamed_file_node = fs_tool.current_dir.contents["file_renamed.txt"]
        if isinstance(renamed_file_node, File):
            self.assertEqual(renamed_file_node.content, "content1")
        else:
            self.fail("file_renamed.txt is not a File")

        mv_result2 = fs_tool.mv(source="dir1", destination="dir_renamed")
        self.assertEqual(mv_result2.get("result"), "Moved 'dir1' to 'dir_renamed'.")
        self.assertNotIn("dir1", fs_tool.current_dir.contents)
        self.assertIn("dir_renamed", fs_tool.current_dir.contents)
        self.assertIsInstance(fs_tool.current_dir.contents["dir_renamed"], Directory)

        fs_tool.current_dir.contents["file_renamed.txt"] = File(
            name="file_renamed.txt", content="content1", parent=fs_tool.current_dir
        )
        mv_result3 = fs_tool.mv(source="file_renamed.txt", destination="dir2")
        self.assertEqual(
            mv_result3.get("result"), "Moved 'file_renamed.txt' into directory 'dir2'."
        )
        self.assertNotIn("file_renamed.txt", fs_tool.current_dir.contents)
        dir2_node_after_mv3 = fs_tool.current_dir.contents["dir2"]
        if isinstance(dir2_node_after_mv3, Directory):
            self.assertIn("file_renamed.txt", dir2_node_after_mv3.contents)
            self.assertEqual(
                dir2_node_after_mv3.contents["file_renamed.txt"].parent,
                dir2_node_after_mv3,
            )
        else:
            self.fail("dir2 is not a Directory after mv3")

        fs_tool.current_dir.contents["dir_renamed"] = Directory(
            name="dir_renamed", parent=fs_tool.current_dir, contents={}
        )
        mv_result4 = fs_tool.mv(source="dir_renamed", destination="dir2")
        self.assertEqual(
            mv_result4.get("result"), "Moved 'dir_renamed' into directory 'dir2'."
        )
        self.assertNotIn("dir_renamed", fs_tool.current_dir.contents)
        dir2_node_after_mv4 = fs_tool.current_dir.contents["dir2"]
        if isinstance(dir2_node_after_mv4, Directory):
            self.assertIn("dir_renamed", dir2_node_after_mv4.contents)
            self.assertEqual(
                dir2_node_after_mv4.contents["dir_renamed"].parent, dir2_node_after_mv4
            )
        else:
            self.fail("dir2 is not a Directory after mv4")

        mv_result5 = fs_tool.mv(source="non_existent.txt", destination="new_name.txt")
        self.assertEqual(
            mv_result5.get("result"), "Error: Source 'non_existent.txt' not found."
        )

        fs_tool.current_dir.contents["another_file.txt"] = File(
            name="another_file.txt",
            content="another content",
            parent=fs_tool.current_dir,
        )
        mv_result6 = fs_tool.mv(
            source="another_file.txt", destination="file_to_overwrite.txt"
        )
        self.assertEqual(
            mv_result6.get("result"),
            "Moved 'another_file.txt' to 'file_to_overwrite.txt'.",
        )
        self.assertNotIn("another_file.txt", fs_tool.current_dir.contents)
        self.assertIn("file_to_overwrite.txt", fs_tool.current_dir.contents)
        overwritten_file_node = fs_tool.current_dir.contents["file_to_overwrite.txt"]
        if isinstance(overwritten_file_node, File):
            self.assertEqual(overwritten_file_node.content, "another content")
        else:
            self.fail("file_to_overwrite.txt is not a File after mv6")

        fs_tool.mkdir("source_dir_for_mv_test7")
        fs_tool.touch("target_file_for_mv_test7.txt")
        mv_result7 = fs_tool.mv(
            source="source_dir_for_mv_test7", destination="target_file_for_mv_test7.txt"
        )
        self.assertEqual(
            mv_result7.get("result"),
            "Error: Cannot overwrite file 'target_file_for_mv_test7.txt' with directory 'source_dir_for_mv_test7'.",
        )

        fs_tool.touch("source_file_for_mv_test8.txt")
        fs_tool.mkdir("target_dir_for_mv_test8")
        mv_result8 = fs_tool.mv(
            source="source_file_for_mv_test8.txt", destination="target_dir_for_mv_test8"
        )
        self.assertEqual(
            mv_result8.get("result"),
            "Moved 'source_file_for_mv_test8.txt' into directory 'target_dir_for_mv_test8'.",
        )
        target_dir_node_after_mv8 = fs_tool.current_dir.contents[
            "target_dir_for_mv_test8"
        ]
        if isinstance(target_dir_node_after_mv8, Directory):
            self.assertIn(
                "source_file_for_mv_test8.txt", target_dir_node_after_mv8.contents
            )
        else:
            self.fail("target_dir_for_mv_test8 is not a Directory after mv8")

        fs_tool.touch("self_move_file.txt")
        mv_result9 = fs_tool.mv(
            source="self_move_file.txt", destination="self_move_file.txt"
        )
        self.assertEqual(
            mv_result9.get("result"),
            "Moved 'self_move_file.txt' to 'self_move_file.txt'.",
        )
        self.assertIn("self_move_file.txt", fs_tool.current_dir.contents)

        fs_tool.touch("path_test_file.txt")
        mv_result10 = fs_tool.mv(
            source="path_test_file.txt", destination="dir2/new_file.txt"
        )
        self.assertEqual(
            mv_result10.get("result"),
            "Error: Destination 'dir2/new_file.txt' cannot be a path.",
        )

        fs_tool.mkdir("mv_dir_self_test")
        mv_result11 = fs_tool.mv(
            source="mv_dir_self_test", destination="mv_dir_self_test"
        )
        self.assertEqual(
            mv_result11.get("result"),
            "Error: Cannot move 'mv_dir_self_test' into itself.",
        )

    def test_gorilla_file_system_sort(self):
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario({})
        unsorted_content = "zebra\napple\nbanana\n"
        fs_tool.current_dir.contents["unsorted.txt"] = File(
            name="unsorted.txt", content=unsorted_content, parent=fs_tool.current_dir
        )
        fs_tool.current_dir.contents["already_sorted.txt"] = File(
            name="already_sorted.txt", content="a\nb\nc", parent=fs_tool.current_dir
        )
        fs_tool.current_dir.contents["empty_lines.txt"] = File(
            name="empty_lines.txt", content="\nc\n\na\nb", parent=fs_tool.current_dir
        )
        fs_tool.current_dir.contents["empty_file.txt"] = File(
            name="empty_file.txt", content="", parent=fs_tool.current_dir
        )
        fs_tool.mkdir("a_directory_for_sort_test")

        sort_result1 = fs_tool.sort(file_name="unsorted.txt")
        self.assertNotIn("error", sort_result1)
        self.assertEqual(sort_result1.get("sorted_content"), "apple\nbanana\nzebra")
        unsorted_node = fs_tool.current_dir.contents["unsorted.txt"]
        if isinstance(unsorted_node, File):
            self.assertEqual(unsorted_node.content, unsorted_content)
        else:
            self.fail("unsorted.txt is not a File")

        sort_result2 = fs_tool.sort(file_name="already_sorted.txt")
        self.assertNotIn("error", sort_result2)
        self.assertEqual(sort_result2.get("sorted_content"), "a\nb\nc")
        already_sorted_node = fs_tool.current_dir.contents["already_sorted.txt"]
        if isinstance(already_sorted_node, File):
            self.assertEqual(already_sorted_node.content, "a\nb\nc")
        else:
            self.fail("already_sorted.txt is not a File")

        sort_result3 = fs_tool.sort(file_name="empty_lines.txt")
        self.assertNotIn("error", sort_result3)
        self.assertEqual(sort_result3.get("sorted_content"), "\n\na\nb\nc")

        sort_result4 = fs_tool.sort(file_name="empty_file.txt")
        self.assertNotIn("error", sort_result4)
        self.assertEqual(sort_result4.get("sorted_content"), "")

        sort_result5 = fs_tool.sort(file_name="no_such_file.txt")
        self.assertIn("error", sort_result5)
        self.assertEqual(
            sort_result5.get("error"),
            "File 'no_such_file.txt' not found or is not a file.",
        )

        sort_result6 = fs_tool.sort(file_name="a_directory_for_sort_test")
        self.assertIn("error", sort_result6)
        self.assertEqual(
            sort_result6.get("error"),
            "File 'a_directory_for_sort_test' not found or is not a file.",
        )

    def test_gorilla_file_system_echo(self):
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario({})
        test_content = "Hello from echo!"
        echo_result1 = fs_tool.echo(content=test_content)
        self.assertNotIn("error", echo_result1)
        self.assertEqual(echo_result1.get("terminal_output"), test_content)
        echo_result2 = fs_tool.echo(content=test_content, file_name=None)
        self.assertNotIn("error", echo_result2)
        self.assertEqual(echo_result2.get("terminal_output"), test_content)
        echo_result2b = fs_tool.echo(content=test_content, file_name="None")
        self.assertNotIn("error", echo_result2b)
        self.assertEqual(echo_result2b.get("terminal_output"), test_content)
        echo_result3 = fs_tool.echo(content=test_content, file_name="echo_new_file.txt")
        self.assertNotIn("error", echo_result3)
        self.assertIsNone(echo_result3.get("terminal_output"))
        self.assertIn("echo_new_file.txt", fs_tool.current_dir.contents)
        new_echo_file_node = fs_tool.current_dir.contents["echo_new_file.txt"]
        if isinstance(new_echo_file_node, File):
            self.assertEqual(new_echo_file_node.content, test_content)
        else:
            self.fail("echo_new_file.txt is not a File")

        fs_tool.current_dir.contents["echo_existing_file.txt"] = File(
            name="echo_existing_file.txt",
            content="original",
            parent=fs_tool.current_dir,
        )
        new_echo_content = "Overwritten by echo!"
        echo_result4 = fs_tool.echo(
            content=new_echo_content, file_name="echo_existing_file.txt"
        )
        self.assertNotIn("error", echo_result4)
        self.assertIsNone(echo_result4.get("terminal_output"))
        existing_echo_file_node = fs_tool.current_dir.contents["echo_existing_file.txt"]
        if isinstance(existing_echo_file_node, File):
            self.assertEqual(existing_echo_file_node.content, new_echo_content)
        else:
            self.fail("echo_existing_file.txt is not a File")

        fs_tool.mkdir("echo_test_dir")
        echo_result5 = fs_tool.echo(content=test_content, file_name="echo_test_dir")
        self.assertIn("error", echo_result5)
        self.assertEqual(
            echo_result5.get("error"),
            "Cannot write to 'echo_test_dir': It is a directory.",
        )
        self.assertIsNone(echo_result5.get("terminal_output"))
        echo_result6 = fs_tool.echo(
            content=test_content, file_name="some_dir/echo_file.txt"
        )
        self.assertIn("error", echo_result6)
        self.assertEqual(
            echo_result6.get("error"),
            "File name 'some_dir/echo_file.txt' cannot be a path.",
        )
        self.assertIsNone(echo_result6.get("terminal_output"))

    def test_gorilla_file_system_wc(self):
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario({})
        content_for_wc = "word1 word2\nanother line with three words\n  leading and trailing spaces  \n\nlast line."
        fs_tool.current_dir.contents["wc_file.txt"] = File(
            name="wc_file.txt", content=content_for_wc, parent=fs_tool.current_dir
        )
        fs_tool.current_dir.contents["wc_empty.txt"] = File(
            name="wc_empty.txt", content="", parent=fs_tool.current_dir
        )
        fs_tool.mkdir("wc_dir_test")
        wc_res1 = fs_tool.wc(file_name="wc_file.txt")
        self.assertNotIn("error", wc_res1)
        self.assertEqual(wc_res1.get("count"), 5)
        self.assertEqual(wc_res1.get("type"), "lines")
        wc_res2 = fs_tool.wc(file_name="wc_file.txt", mode="l")
        self.assertNotIn("error", wc_res2)
        self.assertEqual(wc_res2.get("count"), 5)
        self.assertEqual(wc_res2.get("type"), "lines")
        wc_res3 = fs_tool.wc(file_name="wc_file.txt", mode="w")
        self.assertNotIn("error", wc_res3)
        self.assertEqual(wc_res3.get("count"), 13)
        self.assertEqual(wc_res3.get("type"), "words")
        wc_res4 = fs_tool.wc(file_name="wc_file.txt", mode="c")
        self.assertNotIn("error", wc_res4)
        self.assertEqual(wc_res4.get("count"), len(content_for_wc))
        self.assertEqual(wc_res4.get("type"), "characters")
        wc_res5_l = fs_tool.wc(file_name="wc_empty.txt", mode="l")
        self.assertEqual(wc_res5_l.get("count"), 0)
        self.assertEqual(wc_res5_l.get("type"), "lines")
        wc_res5_w = fs_tool.wc(file_name="wc_empty.txt", mode="w")
        self.assertEqual(wc_res5_w.get("count"), 0)
        self.assertEqual(wc_res5_w.get("type"), "words")
        wc_res5_c = fs_tool.wc(file_name="wc_empty.txt", mode="c")
        self.assertEqual(wc_res5_c.get("count"), 0)
        self.assertEqual(wc_res5_c.get("type"), "characters")
        wc_res6 = fs_tool.wc(file_name="no_wc_file.txt")
        self.assertIn("error", wc_res6)
        self.assertEqual(
            wc_res6.get("error"), "File 'no_wc_file.txt' not found or is not a file."
        )
        wc_res7 = fs_tool.wc(file_name="wc_file.txt", mode="x")  # type: ignore
        self.assertIn("error", wc_res7)
        self.assertEqual(
            wc_res7.get("error"), "Invalid mode 'x'. Must be 'l', 'w', or 'c'."
        )
        wc_res8 = fs_tool.wc(file_name="wc_dir_test")
        self.assertIn("error", wc_res8)
        self.assertEqual(
            wc_res8.get("error"), "File 'wc_dir_test' not found or is not a file."
        )

    def test_gorilla_file_system_tail(self):
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario({})
        lines_15 = "\n".join([f"Line {i+1}" for i in range(15)])
        lines_5 = "\n".join([f"Line {i+1}" for i in range(5)])
        fs_tool.current_dir.contents["file_15_lines.txt"] = File(
            name="file_15_lines.txt", content=lines_15, parent=fs_tool.current_dir
        )
        fs_tool.current_dir.contents["file_5_lines.txt"] = File(
            name="file_5_lines.txt", content=lines_5, parent=fs_tool.current_dir
        )
        fs_tool.current_dir.contents["empty_tail_file.txt"] = File(
            name="empty_tail_file.txt", content="", parent=fs_tool.current_dir
        )
        fs_tool.mkdir("tail_test_dir")
        tail_result1 = fs_tool.tail(file_name="file_15_lines.txt")
        self.assertNotIn("error", tail_result1)
        expected_lines1 = "\n".join([f"Line {i+1}" for i in range(5, 15)])
        self.assertEqual(tail_result1.get("last_lines"), expected_lines1)
        tail_result2 = fs_tool.tail(file_name="file_5_lines.txt")
        self.assertNotIn("error", tail_result2)
        self.assertEqual(tail_result2.get("last_lines"), lines_5)
        tail_result3 = fs_tool.tail(file_name="file_15_lines.txt", lines=3)
        self.assertNotIn("error", tail_result3)
        expected_lines3 = "\n".join([f"Line {i+1}" for i in range(12, 15)])
        self.assertEqual(tail_result3.get("last_lines"), expected_lines3)
        tail_result4 = fs_tool.tail(file_name="file_5_lines.txt", lines=15)
        self.assertNotIn("error", tail_result4)
        self.assertEqual(tail_result4.get("last_lines"), lines_5)
        tail_result5 = fs_tool.tail(file_name="empty_tail_file.txt")
        self.assertNotIn("error", tail_result5)
        self.assertEqual(tail_result5.get("last_lines"), "")
        tail_result6 = fs_tool.tail(file_name="file_15_lines.txt", lines=0)
        self.assertNotIn("error", tail_result6)
        self.assertEqual(tail_result6.get("last_lines"), "")
        tail_result6b = fs_tool.tail(file_name="file_15_lines.txt", lines=-5)
        self.assertNotIn("error", tail_result6b)
        self.assertEqual(tail_result6b.get("last_lines"), "")
        tail_result7 = fs_tool.tail(file_name="no_such_tail_file.txt")
        self.assertIn("error", tail_result7)
        self.assertEqual(
            tail_result7.get("error"),
            "File 'no_such_tail_file.txt' not found or is not a file.",
        )
        tail_result8 = fs_tool.tail(file_name="tail_test_dir")
        self.assertIn("error", tail_result8)
        self.assertEqual(
            tail_result8.get("error"),
            "File 'tail_test_dir' not found or is not a file.",
        )

    def test_gorilla_file_system_find(self):
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario(
            {
                "root": {
                    "type": "directory",
                    "contents": {
                        "file1.txt": {"type": "file", "content": "content1"},
                        "doc.txt": {"type": "file", "content": "document content"},
                        "dir1": {
                            "type": "directory",
                            "contents": {
                                "file_in_dir1.txt": {
                                    "type": "file",
                                    "content": "inside dir1",
                                },
                                "sub_dir": {
                                    "type": "directory",
                                    "contents": {
                                        "nested_doc.txt": {
                                            "type": "file",
                                            "content": "deeply nested",
                                        }
                                    },
                                },
                            },
                        },
                        "dir2": {
                            "type": "directory",
                            "contents": {
                                "another_file.txt": {
                                    "type": "file",
                                    "content": "in dir2",
                                }
                            },
                        },
                    },
                }
            }
        )
        find_res1 = fs_tool.find(path=".")
        self.assertNotIn("error", find_res1)
        self.assertIsInstance(find_res1.get("matches"), list)
        expected1 = sorted(
            [
                "file1.txt",
                "doc.txt",
                "dir1",
                "dir1/file_in_dir1.txt",
                "dir1/sub_dir",
                "dir1/sub_dir/nested_doc.txt",
                "dir2",
                "dir2/another_file.txt",
            ]
        )
        self.assertEqual(
            sorted(find_res1.get("matches") or []), expected1
        )  # Added 'or []'
        find_res2 = fs_tool.find(path=".", name="file1.txt")
        self.assertNotIn("error", find_res2)
        self.assertEqual(sorted(find_res2.get("matches") or []), sorted(["file1.txt"]))
        find_res2b = fs_tool.find(path=".", name="dir1")
        self.assertNotIn("error", find_res2b)
        self.assertEqual(
            sorted(find_res2b.get("matches") or []),
            sorted(["dir1", "dir1/file_in_dir1.txt"]),
        )
        find_res3 = fs_tool.find(path=".", name="doc")
        self.assertNotIn("error", find_res3)
        expected3 = sorted(["doc.txt", "dir1/sub_dir/nested_doc.txt"])
        self.assertEqual(sorted(find_res3.get("matches") or []), expected3)
        find_res4 = fs_tool.find(path="dir1", name=None)
        self.assertNotIn("error", find_res4)
        expected4 = sorted(["file_in_dir1.txt", "sub_dir", "sub_dir/nested_doc.txt"])
        self.assertEqual(sorted(find_res4.get("matches") or []), expected4)
        find_res5 = fs_tool.find(path="dir1", name="file")
        self.assertNotIn("error", find_res5)
        self.assertEqual(
            sorted(find_res5.get("matches") or []), sorted(["file_in_dir1.txt"])
        )
        find_res6 = fs_tool.find(path=".", name="nested_doc.txt")
        self.assertNotIn("error", find_res6)
        self.assertEqual(
            sorted(find_res6.get("matches") or []),
            sorted(["dir1/sub_dir/nested_doc.txt"]),
        )
        find_res7 = fs_tool.find(path="non_existent_dir", name="file1.txt")
        self.assertIn("error", find_res7)
        self.assertEqual(
            find_res7.get("error"),
            "Path 'non_existent_dir' not found or is not a directory.",
        )
        find_res8 = fs_tool.find(path=".", name="ghost_file.boo")
        self.assertNotIn("error", find_res8)
        self.assertEqual(find_res8.get("matches"), [])
        fs_tool.cd(fs_tool.root.name)
        find_res9 = fs_tool.find(path=f"/{fs_tool.root.name}", name="file1.txt")
        self.assertNotIn("error", find_res9, f"Error: {find_res9.get('error')}")
        self.assertEqual(sorted(find_res9.get("matches") or []), sorted(["file1.txt"]))
        find_res9b = fs_tool.find(path=f"/{fs_tool.root.name}", name="nested_doc.txt")
        self.assertNotIn("error", find_res9b, f"Error: {find_res9b.get('error')}")
        self.assertEqual(
            sorted(find_res9b.get("matches") or []),
            sorted(["dir1/sub_dir/nested_doc.txt"]),
        )

    def test_gorilla_file_system_du(self):
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario(
            {
                "root": {
                    "type": "directory",
                    "contents": {
                        "file1.txt": {"type": "file", "content": "0123456789"},
                        "dir1": {
                            "type": "directory",
                            "contents": {
                                "file_in_dir1.txt": {
                                    "type": "file",
                                    "content": "abcde",
                                },
                                "sub_dir": {"type": "directory", "contents": {}},
                            },
                        },
                        "empty_dir": {"type": "directory", "contents": {}},
                    },
                }
            }
        )
        du_res1 = fs_tool.du()
        self.assertNotIn("error", du_res1)
        self.assertEqual(du_res1.get("disk_usage"), "15")
        du_res2 = fs_tool.du(human_readable=True)
        self.assertNotIn("error", du_res2)
        self.assertEqual(du_res2.get("disk_usage"), "15B")
        fs_tool.cd("dir1")
        du_res3 = fs_tool.du()
        self.assertNotIn("error", du_res3)
        self.assertEqual(du_res3.get("disk_usage"), "5")
        du_res3b = fs_tool.du(human_readable=True)
        self.assertNotIn("error", du_res3b)
        self.assertEqual(du_res3b.get("disk_usage"), "5B")
        fs_tool.cd("sub_dir")
        du_res4 = fs_tool.du()
        self.assertNotIn("error", du_res4)
        self.assertEqual(du_res4.get("disk_usage"), "0")
        du_res4b = fs_tool.du(human_readable=True)
        self.assertEqual(du_res4b.get("disk_usage"), "0B")
        fs_tool.cd("..")
        fs_tool.cd("..")
        fs_tool.cd("empty_dir")
        du_res5 = fs_tool.du()
        self.assertNotIn("error", du_res5)
        self.assertEqual(du_res5.get("disk_usage"), "0")
        fs_tool.cd("..")
        fs_tool.mkdir("large_files_dir")
        fs_tool.cd("large_files_dir")
        fs_tool.current_dir.contents["kb_file.txt"] = File(
            name="kb_file.txt", content="a" * 1024, parent=fs_tool.current_dir
        )
        du_res6 = fs_tool.du(human_readable=True)
        self.assertEqual(du_res6.get("disk_usage"), "1.0KB")
        fs_tool.current_dir.contents["mb_file.txt"] = File(
            name="mb_file.txt", content="a" * (1024 * 1024), parent=fs_tool.current_dir
        )
        du_res7 = fs_tool.du(human_readable=True)
        self.assertEqual(du_res7.get("disk_usage"), "1.0MB")


if __name__ == "__main__":
    unittest.main()
