import unittest
from typing import Any, List, Optional

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
        result = math_tool.add(a=5.0, b=3.0)
        self.assertEqual(result.get("result"), 8)
        result_error = math_tool.add(a="test", b=3.0)  # type: ignore[arg-type]
        self.assertIn("error", result_error)

    # Add more tests for MathAPI methods here

    def test_message_api_instantiation_and_load_scenario(self):
        """Test MessageAPI instantiation and _load_scenario."""
        try:
            message_tool = MessageAPI()
            self.assertIsNotNone(message_tool)
            # Basic _load_scenario call with an empty scenario
            # This ensures the method runs and defaults are applied if necessary
            message_tool._load_scenario({})
            # Check if default attributes are set (example)
            self.assertIsNotNone(getattr(message_tool, "user_map", None))
        except Exception as e:
            self.fail(f"MessageAPI instantiation or _load_scenario failed: {e}")

    def test_message_api_list_users(self):
        """Test MessageAPI's list_users method."""
        message_tool = MessageAPI()
        message_tool._load_scenario({})  # Load default state
        result = message_tool.list_users()
        self.assertIn("user_list", result)
        self.assertIsInstance(result["user_list"], list)
        # Based on DEFAULT_STATE in message_api.py
        self.assertEqual(len(result["user_list"]), 4)

    def test_ticket_api_instantiation_and_load_scenario(self):
        """Test TicketAPI instantiation and _load_scenario."""
        try:
            ticket_tool = TicketAPI()
            self.assertIsNotNone(ticket_tool)
            # Basic _load_scenario call
            ticket_tool._load_scenario({})
            self.assertIsNotNone(getattr(ticket_tool, "ticket_queue", None))
        except Exception as e:
            self.fail(f"TicketAPI instantiation or _load_scenario failed: {e}")

    def test_ticket_api_get_ticket(self):
        """Test TicketAPI's get_ticket method."""
        ticket_tool = TicketAPI()
        ticket_tool._load_scenario({})  # Load default state

        # Test getting a non-existent ticket
        result_error = ticket_tool.get_ticket(ticket_id=999)
        self.assertIn("error", result_error)

        # Test creating and getting a ticket (requires user to be "logged in")
        ticket_tool.current_user = "test_user"  # Simulate login
        create_result = ticket_tool.create_ticket(
            title="Test Ticket", description="A test ticket"
        )
        ticket_id = create_result.get("id")
        self.assertIsNotNone(ticket_id)

        if ticket_id is not None and isinstance(ticket_id, (int, str)):
            ticket = ticket_tool.get_ticket(ticket_id=int(ticket_id))
            self.assertEqual(ticket.get("id"), ticket_id)
            self.assertEqual(ticket.get("title"), "Test Ticket")

    def test_trading_bot_instantiation_and_load_scenario(self):
        """Test TradingBot instantiation and _load_scenario."""
        try:
            trading_tool = TradingBot()
            self.assertIsNotNone(trading_tool)
            # Basic _load_scenario call
            trading_tool._load_scenario({})
            self.assertIsNotNone(getattr(trading_tool, "orders", None))
            self.assertIsNotNone(getattr(trading_tool, "stocks", None))
        except Exception as e:
            self.fail(f"TradingBot instantiation or _load_scenario failed: {e}")

    def test_trading_bot_get_stock_info(self):
        """Test TradingBot's get_stock_info method."""
        trading_tool = TradingBot()
        trading_tool._load_scenario({})  # Load default state

        # Test getting info for an existing stock (AAPL is in DEFAULT_STATE)
        result_aapl = trading_tool.get_stock_info(symbol="AAPL")
        self.assertNotIn("error", result_aapl)
        self.assertIn("price", result_aapl)

        # Test getting info for a non-existent stock
        result_error = trading_tool.get_stock_info(symbol="NONEXISTENT")
        self.assertIn("error", result_error)

    def test_travel_api_instantiation_and_load_scenario(self):
        """Test TravelAPI instantiation and _load_scenario."""
        try:
            travel_tool = TravelAPI()
            self.assertIsNotNone(travel_tool)
            # Basic _load_scenario call
            travel_tool._load_scenario({})
            self.assertIsNotNone(getattr(travel_tool, "credit_card_list", None))
            self.assertIsNotNone(getattr(travel_tool, "booking_record", None))
        except Exception as e:
            self.fail(f"TravelAPI instantiation or _load_scenario failed: {e}")

    def test_travel_api_get_flight_cost(self):
        """Test TravelAPI's get_flight_cost method."""
        travel_tool = TravelAPI()
        travel_tool._load_scenario({})  # Load default state

        # Test getting flight cost for a known route
        result = travel_tool.get_flight_cost(
            travel_from="SFO",
            travel_to="LAX",
            travel_date="2024-10-01",
            travel_class="economy",
        )
        self.assertIn("travel_cost_list", result)
        self.assertIsInstance(result["travel_cost_list"], list)
        self.assertTrue(len(result["travel_cost_list"]) > 0)
        self.assertIsInstance(result["travel_cost_list"][0], float)

        # Test with an invalid class
        with self.assertRaises(ValueError) as context:
            travel_tool.get_flight_cost(
                travel_from="SFO",
                travel_to="LAX",
                travel_date="2024-10-01",
                travel_class="invalid_class",
            )
        self.assertIn("Invalid travel class", str(context.exception))

        # Test with an unknown route
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
            # Basic _load_scenario call
            vehicle_tool._load_scenario({})
            self.assertIsNotNone(getattr(vehicle_tool, "fuelLevel", None))
        except Exception as e:
            self.fail(f"VehicleControlAPI instantiation or _load_scenario failed: {e}")

    def test_vehicle_control_api_display_car_status(self):
        """Test VehicleControlAPI's displayCarStatus method."""
        vehicle_tool = VehicleControlAPI()
        vehicle_tool._load_scenario({})  # Load default state

        # Test displaying fuel status
        result_fuel = vehicle_tool.displayCarStatus(option="fuel")
        self.assertIn("fuelLevel", result_fuel)
        self.assertIsInstance(result_fuel["fuelLevel"], float)

        # Test displaying an invalid option
        result_error = vehicle_tool.displayCarStatus(option="invalid_option")
        self.assertIn("error", result_error)

    def test_gorilla_file_system_instantiation_and_load_scenario(self):
        """Test GorillaFileSystem instantiation and _load_scenario."""
        try:
            fs_tool = GorillaFileSystem()
            self.assertIsNotNone(fs_tool)
            # Basic _load_scenario call
            fs_tool._load_scenario({})
            self.assertIsNotNone(
                getattr(fs_tool, "root", None)  # Check for the root directory attribute
            )
        except Exception as e:
            self.fail(f"GorillaFileSystem instantiation or _load_scenario failed: {e}")

    def test_gorilla_file_system_ls_and_cat(self):
        """Test GorillaFileSystem's ls and cat methods."""
        fs_tool = GorillaFileSystem()
        # Load a scenario with a known file, or adapt to create one if possible
        # For now, assume _load_scenario with an empty dict creates a root.
        # The default _load_scenario in GorillaFileSystem creates a 'workspace' root.
        # If it also created a default file, we could test cat.
        # Let's test ls on the root.
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
        self.assertIn("example.txt", ls_result["contents"])
        self.assertEqual(ls_result["contents"]["example.txt"]["type"], "file")

        # Test cat
        cat_result = fs_tool.cat(file_name="example.txt")
        self.assertEqual(cat_result.get("content"), "test content")

        # Test cat on non-existent file
        cat_error_result = fs_tool.cat(file_name="non_existent_file.txt")
        self.assertIn("error", cat_error_result)

    def test_twitter_api_instantiation_and_load_scenario(self):
        """Test TwitterAPI instantiation and _load_scenario."""
        try:
            twitter_tool = TwitterAPI()
            self.assertIsNotNone(twitter_tool)
            # Basic _load_scenario call
            twitter_tool._load_scenario({})
            self.assertIsNotNone(
                getattr(twitter_tool, "tweets", None)
            )  # Check for tweets attribute
            self.assertFalse(twitter_tool.authenticated)  # Should be false by default
        except Exception as e:
            self.fail(f"TwitterAPI instantiation or _load_scenario failed: {e}")

    def test_twitter_api_post_and_get_tweets(self):
        """Test TwitterAPI's post_tweet and get_tweets methods."""
        twitter_tool = TwitterAPI()
        scenario_data = {
            "username": "testuser",
            "password": "password",
            "tweets": {},  # Initial tweets
            "comments": {},
            "retweets": {},
            "following_list": [],
            "tweet_counter": 0,  # Initial counter
            "authenticated": False,
        }
        twitter_tool._load_scenario(scenario_data)

        # Test login
        login_result = twitter_tool.login(username="testuser", password="password")
        self.assertEqual(
            login_result.get("status"),
            "success",
            f"Login failed: {login_result.get('message')}",
        )
        self.assertTrue(twitter_tool.authenticated)

        # Test post_tweet
        tweet_content = "Hello from Cline the AI!"
        post_result = twitter_tool.post_tweet(content=tweet_content)
        self.assertEqual(
            post_result.get("status"),
            "success",
            f"Post tweet failed: {post_result.get('message')}",
        )
        self.assertIn("tweet_id", post_result)
        tweet_id = post_result["tweet_id"]
        tweet_id_str = str(
            tweet_id
        )  # The API stores tweet IDs as strings in the tweets dict

        # Test get_tweets for the user
        tweets_result = twitter_tool.get_tweets(username="testuser")
        self.assertIsInstance(
            tweets_result, dict, "get_tweets should return a dictionary."
        )
        self.assertIn(
            tweet_id_str, tweets_result, "Posted tweet not found in user's tweets dict."
        )
        self.assertEqual(tweets_result[tweet_id_str]["content"], tweet_content)
        self.assertEqual(tweets_result[tweet_id_str]["username"], "testuser")

        # Test get_tweets for all (should include the new tweet)
        all_tweets_result = twitter_tool.get_tweets()
        self.assertIsInstance(all_tweets_result, dict)
        self.assertIn(
            tweet_id_str,
            all_tweets_result,
            "Posted tweet not found in all tweets dict.",
        )
        self.assertEqual(all_tweets_result[tweet_id_str]["content"], tweet_content)

    def test_gorilla_file_system_pwd(self):
        """Test GorillaFileSystem's pwd method."""
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario({})  # Loads default /workspace

        # 1. Test pwd at root
        pwd_result_root = fs_tool.pwd()
        self.assertNotIn(
            "error",
            pwd_result_root,
            f"pwd failed at root: {pwd_result_root.get('error')}",
        )
        self.assertEqual(pwd_result_root.get("current_working_directory"), "/workspace")

        # 2. Create a dir, cd into it, test pwd
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

        # 3. cd back to parent, test pwd
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
        """Test GorillaFileSystem's cp method."""
        fs_tool = GorillaFileSystem()
        # Initial setup: /workspace, /workspace/file1.txt, /workspace/dir1
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

        # 1. Copy file to new name in current directory (/workspace)
        cp_result1 = fs_tool.cp(source="file1.txt", destination="file2.txt")
        self.assertIn("Copied 'file1.txt' to 'file2.txt'", cp_result1.get("result", ""))
        self.assertIn("file2.txt", fs_tool.current_dir.contents)
        self.assertEqual(
            fs_tool.current_dir.contents["file2.txt"].content, "content of file1"
        )

        # 2. Copy file into an existing subdirectory (/workspace/dir1)
        cp_result2 = fs_tool.cp(source="file1.txt", destination="dir1")
        self.assertIn(
            "Copied 'file1.txt' into directory 'dir1'", cp_result2.get("result", "")
        )
        self.assertIn("file1.txt", fs_tool.current_dir.contents["dir1"].contents)
        self.assertEqual(
            fs_tool.current_dir.contents["dir1"].contents["file1.txt"].content,
            "content of file1",
        )

        # 3. Copy directory to new name (deep copy)
        cp_result3 = fs_tool.cp(source="dir1", destination="dir2")
        self.assertIn("Copied 'dir1' to 'dir2'", cp_result3.get("result", ""))
        self.assertIn("dir2", fs_tool.current_dir.contents)
        self.assertIsInstance(
            fs_tool.current_dir.contents["dir2"],
            type(fs_tool.current_dir.contents["dir1"]),
        )  # Check type
        self.assertIn("nested_file.txt", fs_tool.current_dir.contents["dir2"].contents)
        self.assertEqual(
            fs_tool.current_dir.contents["dir2"].contents["nested_file.txt"].content,
            "nested content",
        )
        # Ensure it's a deep copy (modifying copy shouldn't affect original)
        fs_tool.current_dir.contents["dir2"].contents[
            "nested_file.txt"
        ].content = "modified nested"
        self.assertEqual(
            fs_tool.current_dir.contents["dir1"].contents["nested_file.txt"].content,
            "nested content",
        )

        # 4. Copy directory into an existing subdirectory (deep copy)
        fs_tool.mkdir("dir3")
        cp_result4 = fs_tool.cp(source="dir1", destination="dir3")
        self.assertIn(
            "Copied 'dir1' into directory 'dir3'", cp_result4.get("result", "")
        )
        self.assertIn("dir1", fs_tool.current_dir.contents["dir3"].contents)
        self.assertIn(
            "nested_file.txt",
            fs_tool.current_dir.contents["dir3"].contents["dir1"].contents,
        )

        # 5. Attempt to copy non-existent source
        cp_result5 = fs_tool.cp(source="nonexistent.txt", destination="newfile.txt")
        self.assertIn(
            "Error: Source 'nonexistent.txt' not found", cp_result5.get("result", "")
        )

        # 6. Attempt to copy file over an existing directory
        cp_result6 = fs_tool.cp(
            source="file1.txt", destination="dir1"
        )  # dir1 already exists
        # Current implementation overwrites dir1/file1.txt if file1.txt is copied into dir1.
        # If destination is dir1 (the directory itself), it should copy file1.txt into dir1.
        # The test for this specific error "Cannot overwrite directory 'destination' with file 'source'"
        # would require destination to be the name of the directory, not copying *into* it.
        # Let's re-test copying file1.txt to dir1 (as a name, not as a target directory)
        # To do this, dir1 must not exist as a directory for this specific error.
        # So, this test case as written might be testing copy *into* dir, which is allowed.
        # Let's test the specific error: cp file1.txt dir1_as_file_target (where dir1_as_file_target is actually a dir)
        # This is tricky because cp logic first checks if dest is a dir to copy into.
        # The error "Cannot overwrite directory 'destination' with file 'source'"
        # happens if destination is NOT a directory, but an existing item that IS a directory.
        # This means the cp logic: `if destination in self.current_dir.contents and isinstance(self.current_dir.contents[destination], Directory):`
        # handles copying *into* a directory.
        # The error is for `else` block: `if isinstance(self.current_dir.contents[destination], Directory) and isinstance(source_item, File):`
        # This means destination is an existing item, it's a directory, and we are trying to overwrite it with a file.
        # Create a scenario for this:
        fs_tool.mkdir("dir_to_be_overwritten_by_file")
        cp_result6b = fs_tool.cp(
            source="file1.txt", destination="dir_to_be_overwritten_by_file"
        )
        # Corrected assertion: cp file into dir should succeed by copying the file into the dir
        self.assertIn(
            "Copied 'file1.txt' into directory 'dir_to_be_overwritten_by_file'",
            cp_result6b.get("result", ""),
        )
        self.assertIn(
            "file1.txt",
            fs_tool.current_dir.contents["dir_to_be_overwritten_by_file"].contents,
        )
        self.assertEqual(
            fs_tool.current_dir.contents["dir_to_be_overwritten_by_file"]
            .contents["file1.txt"]
            .content,
            "content of file1",
        )

        # 7. Attempt to copy directory over an existing file
        fs_tool.cat = lambda file_name: {
            "content": "dummy"
        }  # mock cat to create a file easily for test
        fs_tool.current_dir.contents[
            "file_to_be_overwritten_by_dir"
        ] = GorillaFileSystem()._deep_copy_item(
            fs_tool.current_dir.contents["file1.txt"],
            fs_tool.current_dir,
            "file_to_be_overwritten_by_dir",
        )

        cp_result7 = fs_tool.cp(
            source="dir1", destination="file_to_be_overwritten_by_dir"
        )
        self.assertIn(
            "Error: Cannot overwrite file 'file_to_be_overwritten_by_dir' with directory 'dir1'",
            cp_result7.get("result", ""),
        )

        # 8. Copy file to an existing file name (overwrite)
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
        self.assertEqual(
            fs_tool.current_dir.contents["file_to_overwrite.txt"].content,
            "content of file1",
        )

        # 9. Copy file into a directory where a file of the same name already exists (overwrite)
        fs_tool.mkdir("dir_for_overwrite_test")
        fs_tool.current_dir.contents["dir_for_overwrite_test"].contents[
            "common_name.txt"
        ] = File(
            name="common_name.txt",
            content="content in dir",
            parent=fs_tool.current_dir.contents["dir_for_overwrite_test"],
        )
        # Create a source file with the same name in current_dir to copy from
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
        self.assertEqual(
            fs_tool.current_dir.contents["dir_for_overwrite_test"]
            .contents["common_name.txt"]
            .content,
            "new main content",
        )

    def test_gorilla_file_system_diff(self):
        """Test GorillaFileSystem's diff method."""
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario({})  # Loads /workspace

        # Setup files for diff
        fs_tool.current_dir.contents["file_a.txt"] = File(
            name="file_a.txt", content="line1\nline2\nline3", parent=fs_tool.current_dir
        )
        fs_tool.current_dir.contents["file_b.txt"] = File(
            name="file_b.txt", content="line1\nline2\nline3", parent=fs_tool.current_dir
        )  # Identical
        fs_tool.current_dir.contents["file_c.txt"] = File(
            name="file_c.txt",
            content="line_one\nline2\nline_three\nline4",
            parent=fs_tool.current_dir,
        )  # Different

        # 1. Compare two identical files
        diff_result1 = fs_tool.diff(file_name1="file_a.txt", file_name2="file_b.txt")
        self.assertEqual(diff_result1.get("status"), "success")
        self.assertEqual(diff_result1.get("diff_lines"), "Files are identical")

        # 2. Compare two different files
        diff_result2 = fs_tool.diff(file_name1="file_a.txt", file_name2="file_c.txt")
        self.assertEqual(diff_result2.get("status"), "success")
        expected_diff_lines = [
            "Line 1: 'line1' != 'line_one'",
            "Line 3: 'line3' != 'line_three'",
            "Line 4: None != 'line4'",
        ]
        self.assertEqual(diff_result2.get("diff_lines"), "\n".join(expected_diff_lines))

        # 2b. Compare files with c first (order might matter for None display)
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

        # 3. Attempt to diff a non-existent file (file1 missing)
        diff_result3 = fs_tool.diff(
            file_name1="non_existent.txt", file_name2="file_a.txt"
        )
        self.assertIn("error", diff_result3)
        self.assertEqual(diff_result3.get("error"), "File non_existent.txt not found")

        # 4. Attempt to diff a non-existent file (file2 missing)
        diff_result4 = fs_tool.diff(
            file_name1="file_a.txt", file_name2="non_existent.txt"
        )
        self.assertIn("error", diff_result4)
        self.assertEqual(diff_result4.get("error"), "File non_existent.txt not found")

    def test_gorilla_file_system_touch(self):
        """Test GorillaFileSystem's touch method."""
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario({})  # Loads /workspace

        # 1. Touch a new file
        touch_result1 = fs_tool.touch(file_name="new_empty_file.txt")
        self.assertEqual(
            touch_result1, {}, "Touch new file should return empty dict on success"
        )
        self.assertIn("new_empty_file.txt", fs_tool.current_dir.contents)
        self.assertIsInstance(fs_tool.current_dir.contents["new_empty_file.txt"], File)
        self.assertEqual(fs_tool.current_dir.contents["new_empty_file.txt"].content, "")

        # 2. Touch an existing file
        fs_tool.current_dir.contents["existing_file.txt"] = File(
            name="existing_file.txt", content="some content", parent=fs_tool.current_dir
        )
        touch_result2 = fs_tool.touch(file_name="existing_file.txt")
        self.assertEqual(
            touch_result2, {}, "Touch existing file should return empty dict on success"
        )
        self.assertEqual(
            fs_tool.current_dir.contents["existing_file.txt"].content,
            "some content",
            "Touch should not alter content of existing file",
        )

        # 3. Attempt to touch an existing directory
        fs_tool.mkdir("existing_dir")
        touch_result3 = fs_tool.touch(file_name="existing_dir")
        self.assertIn("error", touch_result3)
        self.assertEqual(
            touch_result3.get("error"),
            "Cannot touch 'existing_dir': It is a directory.",
        )

    def test_gorilla_file_system_rm(self):
        """Test GorillaFileSystem's rm method."""
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

        # 1. Remove an existing file
        rm_result1 = fs_tool.rm(file_name="file_to_remove.txt")
        self.assertEqual(
            rm_result1.get("result"), "Successfully removed 'file_to_remove.txt'."
        )
        self.assertNotIn("file_to_remove.txt", fs_tool.current_dir.contents)

        # 2. Remove an existing empty directory
        rm_result2 = fs_tool.rm(file_name="empty_dir_to_remove")
        self.assertEqual(
            rm_result2.get("result"), "Successfully removed 'empty_dir_to_remove'."
        )
        self.assertNotIn("empty_dir_to_remove", fs_tool.current_dir.contents)

        # 3. Remove an existing non-empty directory
        rm_result3 = fs_tool.rm(file_name="non_empty_dir_to_remove")
        self.assertEqual(
            rm_result3.get("result"), "Successfully removed 'non_empty_dir_to_remove'."
        )
        self.assertNotIn("non_empty_dir_to_remove", fs_tool.current_dir.contents)

        # 4. Attempt to remove a non-existent file/directory
        rm_result4 = fs_tool.rm(file_name="does_not_exist.txt")
        self.assertEqual(
            rm_result4.get("result"), "Error: 'does_not_exist.txt' not found."
        )

    def test_gorilla_file_system_rmdir(self):
        """Test GorillaFileSystem's rmdir method."""
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

        # 1. Remove an existing empty directory
        rmdir_result1 = fs_tool.rmdir(dir_name="empty_dir")
        self.assertEqual(
            rmdir_result1.get("result"), "Successfully removed directory 'empty_dir'."
        )
        self.assertNotIn("empty_dir", fs_tool.current_dir.contents)

        # 2. Attempt to remove a non-empty directory
        rmdir_result2 = fs_tool.rmdir(dir_name="non_empty_dir")
        self.assertEqual(
            rmdir_result2.get("result"),
            "Error: Directory 'non_empty_dir' is not empty.",
        )
        self.assertIn(
            "non_empty_dir", fs_tool.current_dir.contents
        )  # Should still exist

        # 3. Attempt to remove a file using rmdir
        rmdir_result3 = fs_tool.rmdir(dir_name="a_file.txt")
        self.assertEqual(
            rmdir_result3.get("result"), "Error: 'a_file.txt' is not a directory."
        )
        self.assertIn("a_file.txt", fs_tool.current_dir.contents)  # Should still exist

        # 4. Attempt to remove a non-existent directory
        rmdir_result4 = fs_tool.rmdir(dir_name="ghost_dir")
        self.assertEqual(
            rmdir_result4.get("result"), "Error: Directory 'ghost_dir' not found."
        )

    def test_gorilla_file_system_grep(self):
        """Test GorillaFileSystem's grep method."""
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario({})  # Loads /workspace

        file_content = "Hello world\nThis is a test line\nAnother Test Line with test pattern\nworld wide web"
        fs_tool.current_dir.contents["grep_test_file.txt"] = File(
            name="grep_test_file.txt", content=file_content, parent=fs_tool.current_dir
        )

        # 1. Grep for a pattern that exists
        grep_result1 = fs_tool.grep(file_name="grep_test_file.txt", pattern="world")
        self.assertNotIn("error", grep_result1)
        self.assertEqual(
            grep_result1.get("matching_lines"), ["Hello world", "world wide web"]
        )

        # 2. Grep for a pattern that does not exist
        grep_result2 = fs_tool.grep(
            file_name="grep_test_file.txt", pattern="nonexistentpattern"
        )
        self.assertNotIn("error", grep_result2)
        self.assertEqual(grep_result2.get("matching_lines"), [])

        # 3. Grep in a non-existent file
        grep_result3 = fs_tool.grep(file_name="no_such_file.txt", pattern="world")
        self.assertIn("error", grep_result3)
        self.assertEqual(
            grep_result3.get("error"),
            "File 'no_such_file.txt' not found or is not a file.",
        )

        # 4. Grep for a pattern that matches multiple lines (case sensitive)
        grep_result4 = fs_tool.grep(file_name="grep_test_file.txt", pattern="test")
        self.assertNotIn("error", grep_result4)
        self.assertEqual(
            grep_result4.get("matching_lines"),
            ["This is a test line", "Another Test Line with test pattern"],
        )

        # 5. Grep for a pattern that is case sensitive (no match for "Test" with lowercase "test" pattern)
        # (already covered by above, but explicitly checking for "Test" with "Test" pattern)
        grep_result5 = fs_tool.grep(file_name="grep_test_file.txt", pattern="Test")
        self.assertNotIn("error", grep_result5)
        self.assertEqual(
            grep_result5.get("matching_lines"), ["Another Test Line with test pattern"]
        )

        # 6. Grep for pattern in a file that is actually a directory
        fs_tool.mkdir("grep_dir_test")
        grep_result6 = fs_tool.grep(file_name="grep_dir_test", pattern="world")
        self.assertIn("error", grep_result6)
        self.assertEqual(
            grep_result6.get("error"),
            "File 'grep_dir_test' not found or is not a file.",
        )

    def test_gorilla_file_system_mv(self):
        """Test GorillaFileSystem's mv method."""
        fs_tool = GorillaFileSystem()
        # Setup: /workspace contains file1.txt, dir1 (empty), dir2 (with nested_file.txt)
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

        # 1. Rename a file in current directory
        mv_result1 = fs_tool.mv(source="file1.txt", destination="file_renamed.txt")
        self.assertEqual(
            mv_result1.get("result"), "Moved 'file1.txt' to 'file_renamed.txt'."
        )
        self.assertNotIn("file1.txt", fs_tool.current_dir.contents)
        self.assertIn("file_renamed.txt", fs_tool.current_dir.contents)
        self.assertEqual(
            fs_tool.current_dir.contents["file_renamed.txt"].content, "content1"
        )

        # 2. Rename a directory in current directory
        mv_result2 = fs_tool.mv(source="dir1", destination="dir_renamed")
        self.assertEqual(mv_result2.get("result"), "Moved 'dir1' to 'dir_renamed'.")
        self.assertNotIn("dir1", fs_tool.current_dir.contents)
        self.assertIn("dir_renamed", fs_tool.current_dir.contents)
        self.assertIsInstance(fs_tool.current_dir.contents["dir_renamed"], Directory)

        # 3. Move a file into an existing subdirectory (dir2)
        # Re-create file_renamed.txt as it was moved in test 1
        fs_tool.current_dir.contents["file_renamed.txt"] = File(
            name="file_renamed.txt", content="content1", parent=fs_tool.current_dir
        )
        mv_result3 = fs_tool.mv(source="file_renamed.txt", destination="dir2")
        self.assertEqual(
            mv_result3.get("result"), "Moved 'file_renamed.txt' into directory 'dir2'."
        )
        self.assertNotIn("file_renamed.txt", fs_tool.current_dir.contents)
        self.assertIn("file_renamed.txt", fs_tool.current_dir.contents["dir2"].contents)
        self.assertEqual(
            fs_tool.current_dir.contents["dir2"].contents["file_renamed.txt"].parent,
            fs_tool.current_dir.contents["dir2"],
        )

        # 4. Move a directory into an existing subdirectory (dir2)
        # Re-create dir_renamed as it was moved in test 2
        fs_tool.current_dir.contents["dir_renamed"] = Directory(
            name="dir_renamed", parent=fs_tool.current_dir, contents={}
        )
        mv_result4 = fs_tool.mv(source="dir_renamed", destination="dir2")
        self.assertEqual(
            mv_result4.get("result"), "Moved 'dir_renamed' into directory 'dir2'."
        )
        self.assertNotIn("dir_renamed", fs_tool.current_dir.contents)
        self.assertIn("dir_renamed", fs_tool.current_dir.contents["dir2"].contents)
        self.assertEqual(
            fs_tool.current_dir.contents["dir2"].contents["dir_renamed"].parent,
            fs_tool.current_dir.contents["dir2"],
        )

        # 5. Attempt to move non-existent source
        mv_result5 = fs_tool.mv(source="non_existent.txt", destination="new_name.txt")
        self.assertEqual(
            mv_result5.get("result"), "Error: Source 'non_existent.txt' not found."
        )

        # 6. Overwrite an existing file
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
        self.assertEqual(
            fs_tool.current_dir.contents["file_to_overwrite.txt"].content,
            "another content",
        )

        # 7. Attempt to overwrite file with directory
        fs_tool.mkdir("source_dir_for_mv_test7")
        fs_tool.touch("target_file_for_mv_test7.txt")  # Ensure it's a file
        mv_result7 = fs_tool.mv(
            source="source_dir_for_mv_test7", destination="target_file_for_mv_test7.txt"
        )
        self.assertEqual(
            mv_result7.get("result"),
            "Error: Cannot overwrite file 'target_file_for_mv_test7.txt' with directory 'source_dir_for_mv_test7'.",
        )

        # 8. Attempt to overwrite directory with file (this should move file INTO directory)
        fs_tool.touch("source_file_for_mv_test8.txt")
        fs_tool.mkdir("target_dir_for_mv_test8")
        mv_result8 = fs_tool.mv(
            source="source_file_for_mv_test8.txt", destination="target_dir_for_mv_test8"
        )
        self.assertEqual(
            mv_result8.get("result"),
            "Moved 'source_file_for_mv_test8.txt' into directory 'target_dir_for_mv_test8'.",
        )
        self.assertIn(
            "source_file_for_mv_test8.txt",
            fs_tool.current_dir.contents["target_dir_for_mv_test8"].contents,
        )

        # 9. Move file to itself (should be a no-op or specific message)
        fs_tool.touch("self_move_file.txt")
        mv_result9 = fs_tool.mv(
            source="self_move_file.txt", destination="self_move_file.txt"
        )
        self.assertEqual(
            mv_result9.get("result"),
            "Moved 'self_move_file.txt' to 'self_move_file.txt'.",
        )  # Current logic
        self.assertIn("self_move_file.txt", fs_tool.current_dir.contents)

        # 10. Destination as a path (should fail)
        fs_tool.touch("path_test_file.txt")
        mv_result10 = fs_tool.mv(
            source="path_test_file.txt", destination="dir2/new_file.txt"
        )
        self.assertEqual(
            mv_result10.get("result"),
            "Error: Destination 'dir2/new_file.txt' cannot be a path.",
        )

        # 11. Moving a directory into itself (should fail)
        fs_tool.mkdir("mv_dir_self_test")
        mv_result11 = fs_tool.mv(
            source="mv_dir_self_test", destination="mv_dir_self_test"
        )
        self.assertEqual(
            mv_result11.get("result"),
            "Error: Cannot move 'mv_dir_self_test' into itself.",
        )

    def test_gorilla_file_system_sort(self):
        """Test GorillaFileSystem's sort method."""
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario({})  # Loads /workspace

        # Setup files for sort
        unsorted_content = "zebra\napple\nbanana\n"
        sorted_content_expected = "apple\nbanana\nzebra"  # Note: trailing newline might be handled differently by splitlines/join

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

        # 1. Sort an unsorted file
        sort_result1 = fs_tool.sort(file_name="unsorted.txt")
        self.assertNotIn("error", sort_result1)
        # splitlines() on "text\n" -> ["text"], so "\n".join results in no trailing newline if last line was empty due to \n
        # If unsorted_content = "zebra\napple\nbanana", then splitlines is ['zebra', 'apple', 'banana'] -> sorted ['apple', 'banana', 'zebra'] -> joined "apple\nbanana\nzebra"
        # If unsorted_content = "zebra\napple\nbanana\n", then splitlines() is ['zebra', 'apple', 'banana'].
        # sorted() -> ['apple', 'banana', 'zebra']
        # "\n".join() -> "apple\nbanana\nzebra"
        sorted_content1 = sort_result1.get("sorted_content")
        if sorted_content1 is not None:
            sorted_data = sorted(sorted_content1.splitlines())
            self.assertEqual(sorted_content1, "\n".join(sorted_data))
        else:
            self.assertEqual(sorted_content1, "")
        self.assertEqual(
            fs_tool.current_dir.contents["unsorted.txt"].content,
            unsorted_content,
            "Original file should not be modified.",
        )

        # 2. Sort an already sorted file
        sort_result2 = fs_tool.sort(file_name="already_sorted.txt")
        self.assertNotIn("error", sort_result2)
        sorted_content2 = sort_result2.get("sorted_content")
        self.assertEqual(sorted_content2, "a\nb\nc")
        self.assertEqual(
            fs_tool.current_dir.contents["already_sorted.txt"].content,
            "a\nb\nc",
            "Original file should not be modified.",
        )

        # 3. Sort a file with empty lines
        # content: "\nc\n\na\nb" -> splitlines: ['', 'c', '', 'a', 'b'] -> sorted: ['', '', 'a', 'b', 'c'] -> joined: "\n\na\nb\nc"
        sort_result3 = fs_tool.sort(file_name="empty_lines.txt")
        self.assertNotIn("error", sort_result3)
        sorted_content3 = sort_result3.get("sorted_content")
        self.assertEqual(sorted_content3, "\n\na\nb\nc")

        # 4. Sort an empty file
        sort_result4 = fs_tool.sort(file_name="empty_file.txt")
        self.assertNotIn("error", sort_result4)
        sorted_content4 = sort_result4.get("sorted_content")
        self.assertEqual(sorted_content4, "")

        # 5. Attempt to sort a non-existent file
        sort_result5 = fs_tool.sort(file_name="no_such_file.txt")
        self.assertIn("error", sort_result5)
        self.assertEqual(
            sort_result5.get("error"),
            "File 'no_such_file.txt' not found or is not a file.",
        )

        # 6. Attempt to sort a directory
        sort_result6 = fs_tool.sort(file_name="a_directory_for_sort_test")
        self.assertIn("error", sort_result6)
        self.assertEqual(
            sort_result6.get("error"),
            "File 'a_directory_for_sort_test' not found or is not a file.",
        )

    def test_gorilla_file_system_echo(self):
        """Test GorillaFileSystem's echo method."""
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario({})  # Loads /workspace

        test_content = "Hello from echo!"

        # 1. Echo content to terminal (no file_name)
        echo_result1 = fs_tool.echo(content=test_content)
        self.assertNotIn("error", echo_result1)
        self.assertEqual(echo_result1.get("terminal_output"), test_content)

        # 2. Echo content to terminal (file_name is None explicitly)
        echo_result2 = fs_tool.echo(content=test_content, file_name=None)
        self.assertNotIn("error", echo_result2)
        self.assertEqual(echo_result2.get("terminal_output"), test_content)

        # 2b. Echo content to terminal (file_name is "None" string)
        echo_result2b = fs_tool.echo(content=test_content, file_name="None")
        self.assertNotIn("error", echo_result2b)
        self.assertEqual(echo_result2b.get("terminal_output"), test_content)

        # 3. Echo content to a new file
        echo_result3 = fs_tool.echo(content=test_content, file_name="echo_new_file.txt")
        self.assertNotIn("error", echo_result3)
        self.assertIsNone(echo_result3.get("terminal_output"))
        self.assertIn("echo_new_file.txt", fs_tool.current_dir.contents)
        self.assertEqual(
            fs_tool.current_dir.contents["echo_new_file.txt"].content, test_content
        )

        # 4. Echo content to an existing file (overwrite)
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
        self.assertEqual(
            fs_tool.current_dir.contents["echo_existing_file.txt"].content,
            new_echo_content,
        )

        # 5. Attempt to echo to a file name that is an existing directory
        fs_tool.mkdir("echo_test_dir")
        echo_result5 = fs_tool.echo(content=test_content, file_name="echo_test_dir")
        self.assertIn("error", echo_result5)
        self.assertEqual(
            echo_result5.get("error"),
            "Cannot write to 'echo_test_dir': It is a directory.",
        )
        self.assertIsNone(
            echo_result5.get("terminal_output")
        )  # Ensure terminal_output is None on error if writing to file

        # 6. Attempt to echo to a file name that is a path (should fail)
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
        """Test GorillaFileSystem's wc method."""
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario({})  # Loads /workspace

        content_for_wc = "word1 word2\nanother line with three words\n  leading and trailing spaces  \n\nlast line."
        # Expected:
        # Lines: 5 (splitlines counts empty lines between other lines, and the last line even if no trailing newline)
        # Words: 1 (word1) + 1 (word2) + 1 (another) + 1 (line) + 1 (with) + 1 (three) + 1 (words) + 1 (leading) + 1 (and) + 1 (trailing) + 1 (spaces) + 1 (last) + 1 (line.) = 13
        # Chars: len(content_for_wc)

        fs_tool.current_dir.contents["wc_file.txt"] = File(
            name="wc_file.txt", content=content_for_wc, parent=fs_tool.current_dir
        )
        fs_tool.current_dir.contents["wc_empty.txt"] = File(
            name="wc_empty.txt", content="", parent=fs_tool.current_dir
        )
        fs_tool.mkdir("wc_dir_test")

        # 1. Count lines (default mode)
        wc_res1 = fs_tool.wc(file_name="wc_file.txt")
        self.assertNotIn("error", wc_res1)
        self.assertEqual(
            wc_res1.get("count"), 5
        )  # "word1 word2", "another line with three words", "  leading and trailing spaces  ", "", "last line."
        self.assertEqual(wc_res1.get("type"), "lines")

        # 2. Count lines (explicit 'l' mode)
        wc_res2 = fs_tool.wc(file_name="wc_file.txt", mode="l")
        self.assertNotIn("error", wc_res2)
        self.assertEqual(wc_res2.get("count"), 5)
        self.assertEqual(wc_res2.get("type"), "lines")

        # 3. Count words ('w' mode)
        wc_res3 = fs_tool.wc(file_name="wc_file.txt", mode="w")
        self.assertNotIn("error", wc_res3)
        self.assertEqual(wc_res3.get("count"), 13)
        self.assertEqual(wc_res3.get("type"), "words")

        # 4. Count characters ('c' mode)
        wc_res4 = fs_tool.wc(file_name="wc_file.txt", mode="c")
        self.assertNotIn("error", wc_res4)
        self.assertEqual(wc_res4.get("count"), len(content_for_wc))
        self.assertEqual(wc_res4.get("type"), "characters")

        # 5. wc on an empty file
        wc_res5_l = fs_tool.wc(file_name="wc_empty.txt", mode="l")
        self.assertEqual(wc_res5_l.get("count"), 0)  # splitlines on "" is []
        self.assertEqual(wc_res5_l.get("type"), "lines")

        wc_res5_w = fs_tool.wc(file_name="wc_empty.txt", mode="w")
        self.assertEqual(wc_res5_w.get("count"), 0)
        self.assertEqual(wc_res5_w.get("type"), "words")

        wc_res5_c = fs_tool.wc(file_name="wc_empty.txt", mode="c")
        self.assertEqual(wc_res5_c.get("count"), 0)
        self.assertEqual(wc_res5_c.get("type"), "characters")

        # 6. Attempt wc on a non-existent file
        wc_res6 = fs_tool.wc(file_name="no_wc_file.txt")
        self.assertIn("error", wc_res6)
        self.assertEqual(
            wc_res6.get("error"), "File 'no_wc_file.txt' not found or is not a file."
        )

        # 7. Attempt wc with an invalid mode
        wc_res7 = fs_tool.wc(file_name="wc_file.txt", mode="x")
        self.assertIn("error", wc_res7)
        self.assertEqual(
            wc_res7.get("error"), "Invalid mode 'x'. Must be 'l', 'w', or 'c'."
        )

        # 8. Attempt wc on a directory
        wc_res8 = fs_tool.wc(file_name="wc_dir_test")
        self.assertIn("error", wc_res8)
        self.assertEqual(
            wc_res8.get("error"), "File 'wc_dir_test' not found or is not a file."
        )

    def test_gorilla_file_system_tail(self):
        """Test GorillaFileSystem's tail method."""
        fs_tool = GorillaFileSystem()
        fs_tool._load_scenario({})  # Loads /workspace

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

        # 1. Tail a file with more lines than default (10)
        tail_result1 = fs_tool.tail(file_name="file_15_lines.txt")  # Default lines=10
        self.assertNotIn("error", tail_result1)
        expected_lines1 = "\n".join(
            [f"Line {i+1}" for i in range(5, 15)]
        )  # Lines 6 to 15
        self.assertEqual(tail_result1.get("last_lines"), expected_lines1)

        # 2. Tail a file with fewer lines than default (10)
        tail_result2 = fs_tool.tail(file_name="file_5_lines.txt")  # Default lines=10
        self.assertNotIn("error", tail_result2)
        self.assertEqual(
            tail_result2.get("last_lines"), lines_5
        )  # Should return all 5 lines

        # 3. Tail a file with specific number of lines (3)
        tail_result3 = fs_tool.tail(file_name="file_15_lines.txt", lines=3)
        self.assertNotIn("error", tail_result3)
        expected_lines3 = "\n".join(
            [f"Line {i+1}" for i in range(12, 15)]
        )  # Lines 13, 14, 15
        self.assertEqual(tail_result3.get("last_lines"), expected_lines3)

        # 4. Tail a file with specific number of lines (15), more than available in file_5_lines
        tail_result4 = fs_tool.tail(file_name="file_5_lines.txt", lines=15)
        self.assertNotIn("error", tail_result4)
        self.assertEqual(
            tail_result4.get("last_lines"), lines_5
        )  # Should return all 5 lines

        # 5. Tail an empty file
        tail_result5 = fs_tool.tail(file_name="empty_tail_file.txt")
        self.assertNotIn("error", tail_result5)
        self.assertEqual(tail_result5.get("last_lines"), "")

        # 6. Tail with lines=0
        tail_result6 = fs_tool.tail(file_name="file_15_lines.txt", lines=0)
        self.assertNotIn("error", tail_result6)
        self.assertEqual(tail_result6.get("last_lines"), "")

        # 6b. Tail with lines < 0 (should also be empty)
        tail_result6b = fs_tool.tail(file_name="file_15_lines.txt", lines=-5)
        self.assertNotIn("error", tail_result6b)
        self.assertEqual(tail_result6b.get("last_lines"), "")

        # 7. Attempt to tail a non-existent file
        tail_result7 = fs_tool.tail(file_name="no_such_tail_file.txt")
        self.assertIn("error", tail_result7)
        self.assertEqual(
            tail_result7.get("error"),
            "File 'no_such_tail_file.txt' not found or is not a file.",
        )

        # 8. Attempt to tail a directory
        tail_result8 = fs_tool.tail(file_name="tail_test_dir")
        self.assertIn("error", tail_result8)
        self.assertEqual(
            tail_result8.get("error"),
            "File 'tail_test_dir' not found or is not a file.",
        )

    def test_gorilla_file_system_find(self):
        """Test GorillaFileSystem's find method."""
        fs_tool = GorillaFileSystem()
        # Setup: /workspace contains:
        #   file1.txt
        #   doc.txt
        #   dir1/
        #     file_in_dir1.txt
        #     sub_dir/
        #       nested_doc.txt
        #   dir2/
        #     another_file.txt
        fs_tool._load_scenario(
            {
                "root": {  # workspace
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

        # 1. Find all in current directory (path=".", name=None)
        find_res1 = fs_tool.find(path=".")
        self.assertNotIn("error", find_res1)
        matches1 = find_res1.get("matches")
        self.assertIsInstance(matches1, list)
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
        if matches1 is not None:  # Should be list
            self.assertEqual(sorted(matches1), expected1)
        else:
            self.fail("find_res1.get('matches') returned None")

        # 2. Find items by exact name in current directory (path=".")
        find_res2 = fs_tool.find(path=".", name="file1.txt")
        self.assertNotIn("error", find_res2)
        matches2 = find_res2.get("matches")
        if matches2 is not None:  # Should be list
            self.assertEqual(sorted(matches2), sorted(["file1.txt"]))
        else:
            self.fail("find_res2.get('matches') returned None")

        find_res2b = fs_tool.find(
            path=".", name="dir1"
        )  # dir1 itself and its contents if name matches
        self.assertNotIn("error", find_res2b)
        matches2b = find_res2b.get("matches")
        # Expected: "dir1", and if "dir1" is in "file_in_dir1.txt", that too.
        # Current find logic: if search_name_pattern in name. So "dir1" matches "dir1" and "file_in_dir1.txt"
        if matches2b is not None:  # Should be list
            self.assertEqual(
                sorted(matches2b), sorted(["dir1", "dir1/file_in_dir1.txt"])
            )
        else:
            self.fail("find_res2b.get('matches') returned None")

        # 3. Find items by partial name "doc" in current directory (path=".")
        find_res3 = fs_tool.find(path=".", name="doc")
        self.assertNotIn("error", find_res3)
        matches3 = find_res3.get("matches")
        expected3 = sorted(["doc.txt", "dir1/sub_dir/nested_doc.txt"])
        if matches3 is not None:  # Should be list
            self.assertEqual(sorted(matches3), expected3)
        else:
            self.fail("find_res3.get('matches') returned None")

        # 4. Find all items in a subdirectory (path="dir1", name=None)
        find_res4 = fs_tool.find(path="dir1", name=None)  # Search within dir1
        self.assertNotIn("error", find_res4)
        matches4 = find_res4.get("matches")
        # Paths should be relative to "dir1"
        expected4 = sorted(["file_in_dir1.txt", "sub_dir", "sub_dir/nested_doc.txt"])
        if matches4 is not None:  # Should be list
            self.assertEqual(sorted(matches4), expected4)
        else:
            self.fail("find_res4.get('matches') returned None")

        # 5. Find items by name in a subdirectory (path="dir1", name="file")
        find_res5 = fs_tool.find(path="dir1", name="file")
        self.assertNotIn("error", find_res5)
        matches5 = find_res5.get("matches")
        if matches5 is not None:  # Should be list
            self.assertEqual(sorted(matches5), sorted(["file_in_dir1.txt"]))
        else:
            self.fail("find_res5.get('matches') returned None")

        # 6. Find items recursively (name="nested_doc.txt", path=".")
        find_res6 = fs_tool.find(path=".", name="nested_doc.txt")
        self.assertNotIn("error", find_res6)
        matches6 = find_res6.get("matches")
        if matches6 is not None:  # Should be list
            self.assertEqual(sorted(matches6), sorted(["dir1/sub_dir/nested_doc.txt"]))
        else:
            self.fail("find_res6.get('matches') returned None")

        # 7. Finding with a non-existent path
        find_res7 = fs_tool.find(path="non_existent_dir", name="file1.txt")
        self.assertIn("error", find_res7)
        self.assertEqual(
            find_res7.get("error"),
            "Path 'non_existent_dir' not found or is not a directory.",
        )

        # 8. Finding a name that doesn't exist anywhere
        find_res8 = fs_tool.find(path=".", name="ghost_file.boo")
        self.assertNotIn("error", find_res8)
        matches8 = find_res8.get("matches")
        self.assertEqual(matches8, [])

        # 9. Finding with path as root (e.g. /workspace) - assuming _find_path handles this
        # This test depends on how _find_path and pwd() define absolute paths.
        # If current_dir is /workspace, find(path="/workspace", name="file1.txt")
        # The _find_path will resolve "/workspace" to self.root if current_dir is self.root.
        # The paths returned by _find_recursive are relative to the start_dir_obj.
        # So if start_dir_obj is root, paths are like "file1.txt", "dir1/file_in_dir1.txt"
        fs_tool.cd(fs_tool.root.name)  # Ensure we are at /workspace
        find_res9 = fs_tool.find(path=f"/{fs_tool.root.name}", name="file1.txt")
        self.assertNotIn("error", find_res9, f"Error: {find_res9.get('error')}")
        matches9 = find_res9.get("matches")
        if matches9 is not None:  # Should be list
            self.assertEqual(sorted(matches9), sorted(["file1.txt"]))
        else:
            self.fail("find_res9.get('matches') returned None")

        find_res9b = fs_tool.find(path=f"/{fs_tool.root.name}", name="nested_doc.txt")
        self.assertNotIn("error", find_res9b, f"Error: {find_res9b.get('error')}")
        matches9b = find_res9b.get("matches")
        if matches9b is not None:  # Should be list
            self.assertEqual(sorted(matches9b), sorted(["dir1/sub_dir/nested_doc.txt"]))
        else:
            self.fail("find_res9b.get('matches') returned None")

    def test_gorilla_file_system_du(self):
        """Test GorillaFileSystem's du method."""
        fs_tool = GorillaFileSystem()
        # Setup: /workspace contains:
        #   file1.txt (10 bytes: "0123456789")
        #   dir1/
        #     file_in_dir1.txt (5 bytes: "abcde")
        #     sub_dir/  (empty)
        #   empty_dir/
        fs_tool._load_scenario(
            {
                "root": {  # workspace
                    "type": "directory",
                    "contents": {
                        "file1.txt": {
                            "type": "file",
                            "content": "0123456789",
                        },  # 10 bytes
                        "dir1": {
                            "type": "directory",
                            "contents": {
                                "file_in_dir1.txt": {
                                    "type": "file",
                                    "content": "abcde",
                                },  # 5 bytes
                                "sub_dir": {"type": "directory", "contents": {}},
                            },
                        },
                        "empty_dir": {"type": "directory", "contents": {}},
                    },
                }
            }
        )

        # 1. du on /workspace (human_readable=False) - Total 10 (file1) + 5 (file_in_dir1) = 15 bytes
        # current_dir is /workspace by default after _load_scenario
        du_res1 = fs_tool.du()
        self.assertNotIn("error", du_res1)
        self.assertEqual(du_res1.get("disk_usage"), "15")

        # 2. du on /workspace (human_readable=True)
        du_res2 = fs_tool.du(human_readable=True)
        self.assertNotIn("error", du_res2)
        self.assertEqual(du_res2.get("disk_usage"), "15B")

        # 3. du on /workspace/dir1 (human_readable=False) - Total 5 bytes
        fs_tool.cd("dir1")
        du_res3 = fs_tool.du()
        self.assertNotIn("error", du_res3)
        self.assertEqual(du_res3.get("disk_usage"), "5")

        # 3b. du on /workspace/dir1 (human_readable=True)
        du_res3b = fs_tool.du(human_readable=True)
        self.assertNotIn("error", du_res3b)
        self.assertEqual(du_res3b.get("disk_usage"), "5B")

        # 4. du on /workspace/dir1/sub_dir (empty)
        fs_tool.cd("sub_dir")
        du_res4 = fs_tool.du()
        self.assertNotIn("error", du_res4)
        self.assertEqual(du_res4.get("disk_usage"), "0")
        du_res4b = fs_tool.du(human_readable=True)
        self.assertEqual(du_res4b.get("disk_usage"), "0B")

        # 5. du on /workspace/empty_dir
        fs_tool.cd("..")  # back to dir1
        fs_tool.cd("..")  # back to workspace
        fs_tool.cd("empty_dir")
        du_res5 = fs_tool.du()
        self.assertNotIn("error", du_res5)
        self.assertEqual(du_res5.get("disk_usage"), "0")

        # 6. Test human readable for KB
        fs_tool.cd("..")  # back to workspace
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
        # du in current dir (large_files_dir) will sum kb_file and mb_file
        # Total = 1024 + 1024*1024 = 1024 + 1048576 = 1049600 bytes
        # 1049600 / 1024 = 1025 KB
        # 1025 KB / 1024 = 1.0009765625 MB -> "1.0MB"
        du_res7 = fs_tool.du(human_readable=True)
        self.assertEqual(du_res7.get("disk_usage"), "1.0MB")

    # Add test classes or methods for other APIs below


if __name__ == "__main__":
    unittest.main()
